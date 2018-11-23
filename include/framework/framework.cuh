// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_FRAMEWORK_H
#define HYBRID_FRAMEWORK_H

#include <functional>
#include <map>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <framework/common.h>
#include <framework/variants/api.cuh>
#include <framework/graph_datum.cuh>
#include <framework/variants/common.cuh>
#include <framework/variants/driver.cuh>
#include <framework/hybrid_policy.h>
#include <framework/algo_variants.cuh>
#include <utils/cuda_utils.h>
#include <utils/graphs/traversal.h>
#include <utils/to_json.h>
#include <groute/device/work_source.cuh>
#include "clion_cuda.cuh"

DECLARE_int32(max_iteration);
DECLARE_string(out_wl);
DECLARE_string(lb_push);
DECLARE_string(lb_pull);
DECLARE_double(alpha);
DECLARE_bool(undirected);
DECLARE_bool(wl_sort);
DECLARE_bool(wl_unique);
DECLARE_double(edge_factor);

namespace sepgraph {
    namespace engine {
        using common::Priority;
        using common::LoadBalancing;
        using common::Scheduling;
        using common::Model;
        using common::MsgPassing;
        using common::AlgoVariant;
        using policy::AlgoType;
        using policy::PolicyDecisionMaker;
        using utils::JsonWriter;

        struct Algo {
            static const char *Name() {
                return "Hybrid Graph Engine";
            }
        };

        template<typename TValue, typename TBuffer, typename TWeight, template<typename, typename, typename, typename ...> class TAppImpl, typename... UnusedData>
        class Engine {
        private:
            typedef TAppImpl<TValue, TBuffer, TWeight, UnusedData...> AppImplDeviceObject;
            typedef graphs::GraphDatum<TValue, TBuffer, TWeight> GraphDatum;

            cudaDeviceProp m_dev_props;

            // Graph data
            std::unique_ptr<utils::traversal::Context<Algo>> m_groute_context;
            std::unique_ptr<groute::Stream> m_stream;
            std::unique_ptr<groute::graphs::single::CSRGraphAllocator> m_csr_dev_graph_allocator;
            std::unique_ptr<groute::graphs::single::CSCGraphAllocator> m_csc_dev_graph_allocator;

            std::unique_ptr<AppImplDeviceObject> m_app_inst;

            // App instance
            std::unique_ptr<GraphDatum> m_graph_datum;

            policy::TRunningInfo m_running_info;
            TBuffer current_priority;  //TODO put it into running info

            PolicyDecisionMaker m_policy_decision_maker;

            EngineOptions m_engine_options;


        public:
            Engine(AlgoType algo_type) :
                    m_running_info(algo_type),
                    m_policy_decision_maker(m_running_info) {
                int dev_id = 0;

                GROUTE_CUDA_CHECK(cudaGetDeviceProperties(&m_dev_props, dev_id));
                m_groute_context = std::unique_ptr<utils::traversal::Context<Algo>>
                        (new utils::traversal::Context<Algo>(1));
                m_stream = std::unique_ptr<groute::Stream>(new groute::Stream(dev_id));
            }

            void SetOptions(EngineOptions &engine_options) {
                m_engine_options = engine_options;
            }

            void LoadGraph() {
                Stopwatch sw_load(true);

                groute::graphs::host::CSRGraph &csr_graph = m_groute_context->host_graph;
                LOG("Converting CSR to CSC...\n");
                groute::graphs::host::CSCGraph csc_graph(csr_graph, FLAGS_undirected);

                // Allocate CSR graph on GPU
                m_groute_context->SetDevice(0);
                m_csr_dev_graph_allocator = std::unique_ptr<groute::graphs::single::CSRGraphAllocator>(
                        new groute::graphs::single::CSRGraphAllocator(csr_graph));

                if (FLAGS_undirected) {
                    m_csc_dev_graph_allocator = std::unique_ptr<groute::graphs::single::CSCGraphAllocator>(
                            new groute::graphs::single::CSCGraphAllocator(csc_graph,
                                                                          m_csr_dev_graph_allocator->DeviceObject()));
                } else {
                    m_csc_dev_graph_allocator = std::unique_ptr<groute::graphs::single::CSCGraphAllocator>(
                            new groute::graphs::single::CSCGraphAllocator(csc_graph));
                }

                m_graph_datum = std::unique_ptr<GraphDatum>(new GraphDatum(csr_graph, csc_graph));

                sw_load.stop();

                m_running_info.time_load_graph = sw_load.ms();

                LOG("Load graph time: %f ms (excluded)\n", sw_load.ms());

//                csr_graph.PrintHistogram();
                m_running_info.nnodes = m_groute_context->nvtxs;
                m_running_info.nedges = m_groute_context->nedges;
                m_running_info.total_workload = m_groute_context->nedges * FLAGS_edge_factor;
                current_priority = m_engine_options.GetPriorityThreshold();
            }

            /*
             * Init Graph Value and buffer fields
             */
            void InitGraph(UnusedData &...data) {
                Stopwatch sw_init(true);

                m_app_inst = std::unique_ptr<AppImplDeviceObject>(new AppImplDeviceObject(data...));

                int dev_id = 0;
                groute::Stream &stream = *m_stream;
                GraphDatum &graph_datum = *m_graph_datum;
                const auto &dev_csr_graph = m_csr_dev_graph_allocator->DeviceObject();
                const auto &dev_csc_graph = m_csc_dev_graph_allocator->DeviceObject();
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes);
                dim3 grid_dims, block_dims;

                // Reset worklist
                graph_datum.m_wl_array_in.ResetAsync(stream);
                graph_datum.m_wl_array_out_high.ResetAsync(stream);
                graph_datum.m_wl_array_out_low.ResetAsync(stream);

                // Reset bitmap
                graph_datum.m_wl_middle.ResetAsync(stream);
                graph_datum.m_wl_bitmap_in.ResetAsync(stream);
                graph_datum.m_wl_bitmap_out_high.ResetAsync(stream);
                graph_datum.m_wl_bitmap_middle.ResetAsync(stream);

                m_app_inst->m_csr_graph = dev_csr_graph;
                m_app_inst->m_csc_graph = dev_csc_graph;
                m_app_inst->m_nnodes = graph_datum.nnodes;
                m_app_inst->m_nedges = graph_datum.nedges;
                m_app_inst->m_p_current_round = graph_datum.m_current_round.dev_ptr;

                // Launch kernel to init value/buffer fields
                KernelSizing(grid_dims, block_dims, work_source.get_size());

                auto &app_inst = *m_app_inst;

                kernel::InitGraph
                        << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                        work_source,
                        graph_datum.GetValueDeviceObject(),
                        graph_datum.GetBufferDeviceObject(),
                        graph_datum.GetBufferTmpDeviceObject());

                stream.Sync();

                kernel::RebuildWorklist
                        << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                        work_source,
                        graph_datum.m_wl_array_in.DeviceObject(),
                        graph_datum.GetBufferDeviceObject());
                stream.Sync();

                kernel::InitDegree
                        << < grid_dims, block_dims, 0, stream.cuda_stream >> > (dev_csr_graph,
                        work_source,
                        graph_datum.m_in_degree.dev_ptr,
                        graph_datum.m_out_degree.dev_ptr);

                stream.Sync();

                m_running_info.time_init_graph = sw_init.ms();
                LOG("Init worklist: %u\n", m_graph_datum->m_wl_array_in.GetCount(stream));

                sw_init.stop();

                LOG("InitGraph: %f ms (excluded)\n", sw_init.ms());
            }

            void SaveToJson() {
                JsonWriter &writer = JsonWriter::getInst();

                writer.write("time_input_active_node", m_running_info.time_overhead_input_active_node);
                writer.write("time_output_active_node", m_running_info.time_overhead_output_active_node);
                writer.write("time_input_workload", m_running_info.time_overhead_input_workload);
                writer.write("time_output_workload", m_running_info.time_overhead_output_workload);
                writer.write("time_queue2bitmap", m_running_info.time_overhead_queue2bitmap);
                writer.write("time_bitmap2queue", m_running_info.time_overhead_bitmap2queue);
                writer.write("time_rebuild_worklist", m_running_info.time_overhead_rebuild_worklist);
                writer.write("time_priority_sample", m_running_info.time_overhead_sample);
                writer.write("time_sort_worklist", m_running_info.time_overhead_wl_sort);
                writer.write("time_unique_worklist", m_running_info.time_overhead_wl_unique);
                writer.write("time_kernel", m_running_info.time_kernel);
                writer.write("time_total", m_running_info.time_total);
                writer.write("time_per_round", m_running_info.time_total / m_running_info.current_round);
                writer.write("num_iteration", (int) m_running_info.current_round);

                if (m_engine_options.IsForceVariant()) {
                    writer.write("force_variant", m_engine_options.GetAlgoVariant().ToString());
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                    writer.write("force_push_load_balancing",
                                 LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PUSH)));
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                    writer.write("force_pull_load_balancing",
                                 LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PULL)));
                }

                if (m_engine_options.GetPriorityType() == Priority::NONE) {
                    writer.write("priority_type", "none");
                } else if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                    writer.write("priority_type", "low_high");
                    writer.write("priority_delta", m_engine_options.GetPriorityThreshold());
                } else if (m_engine_options.GetPriorityType() == Priority::SAMPLING) {
                    writer.write("priority_type", "sampling");
                    writer.write("cut_threshold", m_engine_options.GetCutThreshold());
                }

                writer.write("fused_kernel", m_engine_options.IsFused() ? "YES" : "NO");
                writer.write("max_iteration_reached",
                             m_running_info.current_round == FLAGS_max_iteration ? "YES" : "NO");
                //writer.write("date", get_now());
                writer.write("device", m_dev_props.name);
                writer.write("dataset", FLAGS_graphfile);
                writer.write("nnodes", (int) m_graph_datum->nnodes);
                writer.write("nedges", (int) m_graph_datum->nedges);
                writer.write("algo_type", m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME ? "TRAVERSAL_SCHEME"
                                                                                                   : "ITERATIVE_SCHEME");
            }

            void PrintInfo() {
                LOG("--------------Overhead--------------\n");
                LOG("Input active node: %f\n", m_running_info.time_overhead_input_active_node);
                LOG("Output active node: %f\n", m_running_info.time_overhead_output_active_node);
                LOG("Input workload: %f\n", m_running_info.time_overhead_input_workload);
                LOG("Output workload: %f\n", m_running_info.time_overhead_output_workload);
                LOG("Queue2Bitmap: %f\n", m_running_info.time_overhead_queue2bitmap);
                LOG("Bitmap2Queue: %f\n", m_running_info.time_overhead_bitmap2queue);
                LOG("Rebuild worklist: %f\n", m_running_info.time_overhead_rebuild_worklist);
                LOG("Priority sample: %f\n", m_running_info.time_overhead_sample);
                LOG("Sort Worlist: %f\n", m_running_info.time_overhead_wl_sort);
                LOG("Unique Worklist: %f\n", m_running_info.time_overhead_wl_unique);
                LOG("--------------Time statistics---------\n");
                LOG("Kernel time: %f\n", m_running_info.time_kernel);
                LOG("Total time: %f\n", m_running_info.time_total);
                LOG("Total rounds: %d\n", m_running_info.current_round);
                LOG("Time/round: %f\n", m_running_info.time_total / m_running_info.current_round);


                LOG("--------------Engine info-------------\n");
                if (m_engine_options.IsForceVariant()) {
                    LOG("Force variant: %s\n", m_engine_options.GetAlgoVariant().ToString().data());
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                    LOG("Force Push Load balancing: %s\n",
                        LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PUSH)).data());
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                    LOG("Force Pull Load balancing: %s\n",
                        LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PULL)).data());
                }

                if (m_engine_options.GetPriorityType() == Priority::NONE) {
                    LOG("Priority type: NONE\n");
                } else if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                    LOG("Priority type: LOW_HIGH\n");
                    LOG("Priority delta: %f\n", m_engine_options.GetPriorityThreshold());
                } else if (m_engine_options.GetPriorityType() == Priority::SAMPLING) {
                    LOG("Priority type: Sampling\n");
                    LOG("Cut threshold: %f\n", m_engine_options.GetCutThreshold());
                }

                LOG("Fused kernel: %s\n", m_engine_options.IsFused() ? "YES" : "NO");
                LOG("Max iteration reached: %s\n", m_running_info.current_round == FLAGS_max_iteration ? "YES" : "NO");


                LOG("-------------Misc-------------------\n");
                //LOG("Date: %s\n", get_now().data());
                LOG("Device: %s\n", m_dev_props.name);
                LOG("Dataset: %s\n", FLAGS_graphfile.data());
                LOG("Algo type: %s\n",
                    m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME ? "TRAVERSAL_SCHEME" : "ITERATIVE_SCHEME");
            }

            void Start() {
                AlgoVariant next_policy = m_policy_decision_maker.GetInitPolicy();
                bool convergence = false;
                std::vector<std::string> round_info;


                Stopwatch sw_total(true);

                LoadOptions();

                while (!convergence) {
                    if (m_engine_options.IsForceVariant()) {
                        next_policy = m_engine_options.GetAlgoVariant();
                    }

                    PreComputation(next_policy);
                    ExecutePolicy(next_policy);
                    PostComputation(next_policy);

                    if (FLAGS_trace) {
                        std::string s_round = format(
                                "Round: %d Policy to execute: %s Time: %f In-nodes: %u Out-nodes: %u Input-edges: %u Output-edges: %u Total-workload: %u",
                                m_running_info.current_round,
                                next_policy.ToString().data(),
                                m_running_info.policy_time[next_policy],
                                m_running_info.input_active_count,
                                m_running_info.output_active_count,
                                m_running_info.input_workload,
                                m_running_info.output_workload,
                                m_running_info.total_workload);

                        round_info.push_back(s_round);
                        printf("%s\n", s_round.data());
                    }

                    if (!m_engine_options.IsForceVariant()) {
                        next_policy = m_policy_decision_maker.GetNextPolicy();
                    }

                    if (FLAGS_trace) {
                        //m_policy_decision_maker.PrintInfo();
                    }

                    // Convergence Check
                    convergence = m_running_info.output_active_count == 0;

                    if (m_running_info.current_round == FLAGS_max_iteration) {
                        convergence = true;
                        LOG("Max iterations reached\n");
                    }
                }

                sw_total.stop();

                m_running_info.time_total = sw_total.ms();

                if (FLAGS_trace) {
                    JsonWriter::getInst().write("round", round_info);
                }
                SaveToJson();
            }

            const groute::graphs::host::CSRGraph &CSRGraph() const {
                return m_groute_context->host_graph;
            }

            const GraphDatum &GetGraphDatum() const {
                return *m_graph_datum;
            }

            const std::vector<TValue> &GatherValue() {
                return m_graph_datum->GatherValue();
            }

            const std::vector<TValue> &GatherBuffer() {
                return m_graph_datum->GatherBuffer();
            }

            groute::graphs::dev::CSRGraph CSRDeviceObject() const {
                return m_csr_dev_graph_allocator->DeviceObject();
            }

            const groute::Stream &getStream() const {
                return *m_stream;
            }

        private:
            void LoadOptions() {
                if (!m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                    if (FLAGS_lb_push.size() == 0) {
                        if (m_groute_context->host_graph.avg_degree() >= 5) {
                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::FINE_GRAINED);
                        }
                    } else {
                        if (FLAGS_lb_push == "none") {
                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::NONE);
                        } else if (FLAGS_lb_push == "coarse") {
                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::COARSE_GRAINED);
                        } else if (FLAGS_lb_push == "fine") {
                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::FINE_GRAINED);
                        } else if (FLAGS_lb_push == "hybrid") {
                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::HYBRID);
                        } else {
                            fprintf(stderr, "unknown push load-balancing policy");
                            exit(1);
                        }
                    }
                }

                if (!m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                    if (FLAGS_lb_pull.size() == 0) {
                        if (m_groute_context->host_graph.avg_degree() >= 5) {
                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::FINE_GRAINED);
                        }
                    } else {
                        if (FLAGS_lb_pull == "none") {
                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::NONE);
                        } else if (FLAGS_lb_pull == "coarse") {
                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::COARSE_GRAINED);
                        } else if (FLAGS_lb_pull == "fine") {
                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::FINE_GRAINED);
                        } else if (FLAGS_lb_pull == "hybrid") {
                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::HYBRID);
                        } else {
                            fprintf(stderr, "unknown pull load-balancing policy");
                            exit(1);
                        }
                    }
                }

                if (FLAGS_alpha == 0) {
                    fprintf(stderr, "Warning: alpha = 0, A general method AsyncPushDD is used\n");
                    m_engine_options.ForceVariant(AlgoVariant::ASYNC_PUSH_DD);
                }
            }

            void PreComputation(const AlgoVariant &algo_variant) {
                const int dev_id = 0;
                const groute::Stream &stream = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                dim3 grid_dims, block_dims;

                // For here, increase the round
                m_running_info.current_round++;
                graph_datum.m_current_round.set_val_H2DAsync(m_running_info.current_round, stream.cuda_stream);

                // Bitmap <--> Worklist
                if (algo_variant.m_scheduling == Scheduling::DATA_DRIVEN) {
                    if (algo_variant.GetMsgPassing() == MsgPassing::PUSH) {
                        // Last round is pull+dd, bitmap -> queue
                        if (m_running_info.current_round > 1 &&
                            m_running_info.last_variant.GetScheduling() == Scheduling::DATA_DRIVEN &&
                            m_running_info.last_variant.GetMsgPassing() == MsgPassing::PULL) {
                            Stopwatch sw_overhead(true);
                            if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                                Bitmap2Queue(graph_datum.m_wl_bitmap_in, graph_datum.m_wl_array_in, stream);

                                if (graph_datum.m_wl_bitmap_out_low.GetPositiveCount(stream) > 0) {
                                    Bitmap2Queue(graph_datum.m_wl_bitmap_out_low, graph_datum.m_wl_array_out_low,
                                                 stream);
                                } else {
                                    Bitmap2Queue(graph_datum.m_wl_bitmap_out_high, graph_datum.m_wl_array_out_high,
                                                 stream);
                                }
                            } else {
                                Bitmap2Queue(graph_datum.m_wl_bitmap_in, graph_datum.m_wl_array_in, stream);
                            }


                            sw_overhead.stop();
                            m_running_info.time_overhead_bitmap2queue += sw_overhead.ms();
                        }

                        if (FLAGS_wl_sort) {
                            uint32_t wl_size = graph_datum.m_wl_array_in.GetCount(stream);
                            Stopwatch sw_sort(true);
                            thrust::device_ptr<index_t> p_worklist(graph_datum.m_wl_array_in.GetDeviceDataPtr());
                            thrust::sort(thrust::cuda::par.on(stream.cuda_stream), p_worklist, p_worklist + wl_size);
                            sw_sort.stop();

                            m_running_info.time_overhead_wl_sort += sw_sort.ms();
                        }
                    } else if (algo_variant.GetMsgPassing() == MsgPassing::PULL) {
                        if (m_running_info.current_round == 1 ||
                            (m_running_info.last_variant.GetScheduling() == Scheduling::DATA_DRIVEN &&
                             m_running_info.last_variant.GetMsgPassing() == MsgPassing::PUSH)) {
                            Stopwatch sw_overhead(true);


                            if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                                Queue2Bitmap(graph_datum.m_wl_array_in,
                                             graph_datum.m_wl_bitmap_in,
                                             stream);

                                if (graph_datum.m_wl_array_out_low.GetCount(stream) > 0) {
                                    Queue2Bitmap(graph_datum.m_wl_array_out_low,
                                                 graph_datum.m_wl_bitmap_out_low,
                                                 stream);
                                } else {
                                    Queue2Bitmap(graph_datum.m_wl_array_out_high,
                                                 graph_datum.m_wl_bitmap_out_high,
                                                 stream);
                                }
                            } else {
                                Queue2Bitmap(graph_datum.m_wl_array_in, graph_datum.m_wl_bitmap_in, stream);
                            }

                            sw_overhead.stop();
                            m_running_info.time_overhead_queue2bitmap += sw_overhead.ms();
                        }
                    }
                }

                // If force variant, don't do statistics job
                if (FLAGS_trace || !m_engine_options.IsForceVariant()) {
                    // Estimate workload
                    if (algo_variant.GetScheduling() == Scheduling::DATA_DRIVEN) {
                        if (algo_variant.m_msg_passing == MsgPassing::PUSH) {
                            Stopwatch sw_input_active_node(true);
                            m_running_info.input_active_count = graph_datum.m_wl_array_in.GetCount(stream);
                            sw_input_active_node.stop();
                            m_running_info.time_overhead_input_active_node += sw_input_active_node.ms();

                            Stopwatch sw_input_workload(true);
                            graph_datum.m_total_out_degree.set_val_H2DAsync(0, stream.cuda_stream);
                            KernelSizing(grid_dims, block_dims, m_running_info.input_active_count);

                            kernel::SumOutDegreeQueue
                                    << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                   (groute::dev::WorkSourceArray<index_t>(
                                                                           graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                                                           m_running_info.input_active_count),
                                                                           graph_datum.m_out_degree.dev_ptr,
                                                                           graph_datum.m_total_out_degree.dev_ptr);
                            stream.Sync();
                            m_running_info.input_workload = graph_datum.m_total_out_degree.get_val_D2H(); // input workload means scout count.
                            sw_input_workload.stop();

                            m_running_info.time_overhead_input_workload += sw_input_workload.ms();
                        } else if (algo_variant.m_msg_passing == MsgPassing::PULL) {
                            Stopwatch sw_input_node(true);
                            m_running_info.input_workload = 0;
                            m_running_info.input_active_count = graph_datum.m_wl_bitmap_in.GetPositiveCount(stream);
                            sw_input_node.stop();
                            m_running_info.time_overhead_input_active_node += sw_input_node.ms();
                        }

                        m_running_info.total_workload -= m_running_info.input_workload;
                    } else {
                        // For TD model, all nodes are active node
                        m_running_info.input_workload = graph_datum.nedges;
                        m_running_info.input_active_count = graph_datum.nnodes;
                    }
                }

                stream.Sync();
            }

            void ExecutePolicy(const AlgoVariant &algo_variant) {
                auto &app_inst = *m_app_inst;
                const int dev_id = 0;
                auto csr_graph = m_csr_dev_graph_allocator->DeviceObject();
                auto csc_graph = m_csc_dev_graph_allocator->DeviceObject();
                const groute::Stream &stream = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                float time_sample = 0;

                Stopwatch sw_execution(true);

                app_inst.SetVariant(algo_variant);

                if (algo_variant == AlgoVariant::SYNC_PUSH_TD) {
                    assert(false);
                } else if (algo_variant == AlgoVariant::SYNC_PUSH_DD) {
                    assert(false);
                } else if (algo_variant == AlgoVariant::SYNC_PULL_TD) {
                    RunSyncPullTD(app_inst,
                                  csc_graph,
                                  graph_datum,
                                  m_engine_options,
                                  stream);
                } else if (algo_variant == AlgoVariant::SYNC_PULL_DD) {
                    RunSyncPullDD(app_inst,
                                  csc_graph,
                                  graph_datum,
                                  m_engine_options,
                                  stream);
                } else if (algo_variant == AlgoVariant::ASYNC_PUSH_TD) {
                    if (m_engine_options.IsFused()) {
                        RunAsyncPushTDFused<AppImplDeviceObject, groute::graphs::dev::CSRGraph,
                                GraphDatum, TValue, TBuffer, TWeight>(app_inst,
                                                                      csr_graph,
                                                                      graph_datum,
                                                                      m_engine_options,
                                                                      m_dev_props,
                                                                      stream);
                    } else {
                        if (m_engine_options.GetPriorityType() == Priority::SAMPLING) {
                            Stopwatch sw_sample(true);
                            TBuffer priority_threshold = GetPriorityThreshold<GraphDatum, TBuffer>(graph_datum,
                                                                                                   m_engine_options.GetCutThreshold(),
                                                                                                   stream);
                            sw_sample.stop();
                            time_sample = sw_sample.ms();

                            RunAsyncPushTDPrio(app_inst,
                                               csr_graph,
                                               graph_datum,
                                               m_engine_options,
                                               priority_threshold,
                                               stream);
                        } else {
                            RunAsyncPushTD(app_inst,
                                           csr_graph,
                                           graph_datum,
                                           m_engine_options,
                                           stream);
                        }
                    }
                } else if (algo_variant == AlgoVariant::ASYNC_PUSH_DD) {
                    if (m_engine_options.IsFused()) {
                        if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                            RunAsyncPushDDFusedPrio<AppImplDeviceObject, groute::graphs::dev::CSRGraph,
                                    GraphDatum, TValue, TBuffer, TWeight>(
                                    app_inst,
                                    csr_graph,
                                    graph_datum,
                                    m_engine_options,
                                    m_dev_props,
                                    stream);
                        } else {
                            RunAsyncPushDDFused<AppImplDeviceObject, groute::graphs::dev::CSRGraph,
                                    GraphDatum, TValue, TBuffer, TWeight>(
                                    app_inst,
                                    csr_graph,
                                    graph_datum,
                                    m_engine_options,
                                    m_dev_props,
                                    stream);
                        }
                    } else {
                        if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                            RunAsyncPushDDPrio<AppImplDeviceObject, TBuffer, groute::graphs::dev::CSRGraph,
                                    GraphDatum>(app_inst,
                                                csr_graph,
                                                current_priority,
                                                graph_datum,
                                                m_engine_options,
                                                stream);
                        } else {
                            RunAsyncPushDD(app_inst,
                                           csr_graph,
                                           graph_datum,
                                           m_engine_options,
                                           stream);
                        }
                    }
                } else if (algo_variant == AlgoVariant::ASYNC_PULL_TD) {
                    assert(false);
                } else if (algo_variant == AlgoVariant::ASYNC_PULL_DD) {
                    if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                        RunAsyncPullDDPrio<AppImplDeviceObject, TBuffer, groute::graphs::dev::CSCGraph,
                                GraphDatum>(app_inst,
                                            csc_graph,
                                            current_priority,
                                            graph_datum,
                                            m_engine_options,
                                            stream);
                    } else {
                        RunAsyncPullDD<AppImplDeviceObject, groute::graphs::dev::CSCGraph,
                                GraphDatum>(app_inst,
                                            csc_graph,
                                            graph_datum,
                                            m_engine_options,
                                            stream);
                    }
                }
                sw_execution.stop();

                m_running_info.time_overhead_sample += time_sample;
                m_running_info.policy_time[algo_variant] = sw_execution.ms() - time_sample;
                m_running_info.time_kernel += sw_execution.ms() - time_sample;
                m_running_info.last_variant = algo_variant;
            }

            void PostComputation(const AlgoVariant &algo_variant) {
                int dev_id = 0;
                const groute::Stream &stream = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;

                kernel::CallPostComputation << < 1, 1, 0, stream.cuda_stream >> > (app_inst);
                stream.Sync();

                m_running_info.current_round = m_graph_datum->m_current_round.get_val_D2H();

                if (algo_variant.m_scheduling == Scheduling::TOPOLOGY_DRIVEN) {
                    Stopwatch sw_rebuild_wl(true);
                    RebuildArrayWorklist(app_inst,
                                         graph_datum,
                                         stream);
                    stream.Sync();
                    sw_rebuild_wl.stop();
                    m_running_info.time_overhead_rebuild_worklist += sw_rebuild_wl.ms();

                    Stopwatch sw_output_active_node(true);
                    m_running_info.output_active_count = graph_datum.m_wl_array_in.GetCount(stream);
                    stream.Sync();
                    sw_output_active_node.stop();
                    m_running_info.time_overhead_output_active_node += sw_output_active_node.ms();
                } else if (algo_variant.m_scheduling == Scheduling::DATA_DRIVEN) {
                    if (algo_variant.GetMsgPassing() == MsgPassing::PUSH) {
                        if (FLAGS_wl_unique) {
                            Stopwatch sw_unique(true);
                            Queue2Bitmap(graph_datum.m_wl_array_in,
                                         graph_datum.m_wl_bitmap_middle,
                                         stream);

                            Bitmap2Queue(graph_datum.m_wl_bitmap_middle,
                                         graph_datum.m_wl_array_in,
                                         stream);
                            stream.Sync();
                            sw_unique.stop();

                            m_running_info.time_overhead_wl_unique += sw_unique.ms();
                        }

                        Stopwatch sw_output_active_node(true);
                        m_running_info.output_active_count = graph_datum.m_wl_array_in.GetCount(stream);
                        sw_output_active_node.stop();
                        m_running_info.time_overhead_output_active_node += sw_output_active_node.ms();
                    } else if (algo_variant.GetMsgPassing() == MsgPassing::PULL) {
                        Stopwatch sw_output_active_node(true);
                        m_running_info.output_active_count = graph_datum.m_wl_bitmap_in.GetPositiveCount(stream);
                        sw_output_active_node.stop();
                        m_running_info.time_overhead_output_active_node += sw_output_active_node.ms();
                    }
                    stream.Sync();
                }

                if (FLAGS_trace || !m_engine_options.IsForceVariant()) {
                    Stopwatch sw_output_workload(true);
                    // TD or DD & PUSH
                    if (algo_variant.m_scheduling == Scheduling::TOPOLOGY_DRIVEN ||
                        algo_variant.m_msg_passing == MsgPassing::PUSH) {
                        uint32_t work_size = graph_datum.m_wl_array_in.GetCount(stream);
                        dim3 grid_dims, block_dims;

                        graph_datum.m_total_out_degree.set_val_H2DAsync(0, stream.cuda_stream);
                        KernelSizing(grid_dims, block_dims, work_size);

                        kernel::SumOutDegreeQueue << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                                 (groute::dev::WorkSourceArray<index_t>(
                                                                                         graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                                                                         work_size),
                                                                                         graph_datum.m_out_degree.dev_ptr,
                                                                                         graph_datum.m_total_out_degree.dev_ptr);
                        stream.Sync();
                        m_running_info.output_workload = graph_datum.m_total_out_degree.get_val_D2H();
                    } else {
                        // DD & pull
                        if (FLAGS_trace) {
                            dim3 grid_dims, block_dims;

                            graph_datum.m_total_out_degree.set_val_H2DAsync(0, stream.cuda_stream);
                            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);

                            kernel::SumOutDegreeBitmap << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                                      (graph_datum.m_wl_bitmap_in.DeviceObject(),
                                                                                              graph_datum.m_out_degree.dev_ptr,
                                                                                              graph_datum.m_total_out_degree.dev_ptr);
                            stream.Sync();
                            printf("output edges: %u\n", graph_datum.m_total_out_degree.get_val_D2H());
//                            m_running_info.output_workload = graph_datum.m_total_out_degree.get_val_D2H();
                        }
                        m_running_info.output_workload = 1;
                    }

                    sw_output_workload.stop();
                    m_running_info.time_overhead_output_workload += sw_output_workload.ms();
                }
            }
        };


    }
}

#endif //HYBRID_FRAMEWORK_H
