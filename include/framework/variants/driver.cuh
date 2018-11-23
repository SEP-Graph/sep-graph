// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_DRIVER_H
#define HYBRID_DRIVER_H

#include <cub/grid/grid_barrier.cuh>
#include <groute/device/work_source.cuh>
#include <groute/graphs/csr_graph.h>
#include <utils/cuda_utils.h>
#include <groute/device/bitmap_impls.h>
#include <framework/variants/api.cuh>
#include <framework/common.h>
#include <framework/variants/async_push_td.cuh>
#include <framework/variants/async_push_dd.cuh>
#include <framework/variants/async_pull_dd.cuh>
#include <framework/variants/async_pull_td.cuh>
#include <framework/variants/sync_pull_td.cuh>
#include <framework/variants/sync_pull_dd.cuh>
#include <framework/clion_cuda.cuh>

namespace sepgraph {
    namespace kernel {
        using sepgraph::common::LoadBalancing;

        template<typename WorkSource,
                typename TValue,
                typename TBuffer>
        __global__
        void PrintTable(WorkSource work_source,
                        groute::graphs::dev::GraphDatum<TValue> node_value_datum,
                        groute::graphs::dev::GraphDatum<TBuffer> node_buffer_datum) {
            const uint32_t tid = TID_1D;
            const uint32_t nthreads = TOTAL_THREADS_1D;
            const uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                printf("%u %u %u\n", node, node_value_datum[i], node_buffer_datum[i]);
            }
        }


        template<typename WorkSource>
        __global__
        void PrintDegree(WorkSource work_source,
                         uint32_t *p_in_degree,
                         uint32_t *p_out_degree) {
            uint32_t tid = TID_1D;
            uint32_t work_size = work_source.get_size();

            if (tid == 0) {
                for (int i = 0; i < work_size; i++) {
                    index_t node = work_source.get_work(i);

                    printf("node: %u in-degree: %u out-degree: %u\n",
                           node,
                           p_in_degree[node],
                           p_out_degree[node]);
                }
            }
        }

        // TODO ALL replace get_item with operator[]
        template<typename WorkSource,
                typename TValue,
                typename TBuffer,
                typename TWeight,
                template<typename, typename, typename, typename ...> class TAppImpl,
                typename... UnusedData>
        __global__
        void InitGraph(TAppImpl<TValue, TBuffer, TWeight, UnusedData...> app_inst,
                       WorkSource work_source,
                       groute::graphs::dev::GraphDatum<TValue> node_value_datum,
                       groute::graphs::dev::GraphDatum<TBuffer> node_buffer_datum,
                       groute::graphs::dev::GraphDatum<TBuffer> node_tmp_buffer_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                TBuffer init_buffer = app_inst.GetInitBuffer(node);

                node_value_datum[node] = app_inst.GetInitValue(node);
                node_buffer_datum[node] = init_buffer;
                node_tmp_buffer_datum[node] = init_buffer;
            }
        }

        template<typename CSRGraph, typename WorkSource>
        __global__ void InitDegree(CSRGraph csr_graph,
                                   WorkSource work_source,
                                   uint32_t *p_in_degree,
                                   uint32_t *p_out_degree) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                index_t begin_edge = csr_graph.begin_edge(node),
                        end_edge = csr_graph.end_edge(node);

                p_out_degree[node] = end_edge - begin_edge;

                for (int edge = begin_edge; edge < end_edge; edge++) {
                    index_t dest = csr_graph.edge_dest(edge);

                    atomicAdd(&p_in_degree[dest], 1);
                }
            }
        }


        template<typename TQueue, typename TBitmap>
        __global__ void QueueToBitmap(TQueue queue, TBitmap bitmap) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = queue.count();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                bitmap.set_bit_atomic(queue.read(i));
            }
        }

        template<typename TBitmap, typename TQueue>
        __global__ void BitmapToQueue(TBitmap bitmap, TQueue queue) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = bitmap.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                if (bitmap.get_bit(i)) {
                    queue.append(i);
                }
            }
        }

        template<typename WorkSource, typename TBuffer>
        __global__ void Sample(WorkSource work_source,
                               groute::graphs::dev::GraphDatum<TBuffer> node_buffer_datum,
                               TBuffer *p_sampled_values) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);

                p_sampled_values[i] = node_buffer_datum.get_item(node);
            }
        }

        template<typename TAppInst,
                typename WorkSource,
                template<typename> class GraphDatum,
                typename TBuffer>
        __global__
        void RebuildWorklist(TAppInst app_inst,
                             WorkSource work_source,
                             groute::dev::Queue<index_t> work_target,
                             GraphDatum<TBuffer> node_buffer_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);

                if (app_inst.IsActiveNode(node, node_buffer_datum.get_item(node))) {
                    work_target.append(node);
                }
            }
        }

        template<typename TAppInst,
                typename WorkSource,
                typename TBitmap,
                template<typename> class GraphDatum,
                typename TBuffer>
        __global__
        void RebuildWorklist(TAppInst app_inst,
                             WorkSource work_source,
                             TBitmap work_target,
                             GraphDatum<TBuffer> node_buffer_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);

                if (app_inst.IsActiveNode(node, node_buffer_datum.get_item(node))) {
                    work_target.set_bit_atomic(node);
                }
            }
        }

        template<typename TAppInst,
                typename WorkSource,
                template<typename> class GraphDatum,
                typename TBuffer>
        __global__
        void RebuildWorklistQ(TAppInst app_inst,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target,
                              GraphDatum<TBuffer> node_buffer_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);

                if (app_inst.IsActiveNode(node, node_buffer_datum.get_item(node))) {
                    work_target.append(node);
                }
            }
        }

        template<typename WorkSource>
        __global__ void SumOutDegreeQueue(WorkSource work_source,
                                     uint32_t *p_out_degree,
                                     uint32_t *p_total_out_degree) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            uint32_t local_sum = 0;
            typedef cub::WarpReduce<uint32_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage temp_storage[8];

            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);

                local_sum += p_out_degree[node];
            }

            int warp_id = threadIdx.x / 32;
            int aggregate = WarpReduce(temp_storage[warp_id]).Sum(local_sum);

            if (cub::LaneId() == 0) {
                atomicAdd(p_total_out_degree, aggregate);
            }
        }

        template <typename TBitmap>
        __global__ void SumOutDegreeBitmap(TBitmap work_source,
                                     uint32_t *p_out_degree,
                                     uint32_t *p_total_out_degree) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            uint32_t local_sum = 0;
            typedef cub::WarpReduce<uint32_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage temp_storage[8];


            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = i;

                if (work_source.get_bit(node)) {
                    local_sum += p_out_degree[node];
                }
            }

            int warp_id = threadIdx.x / 32;
            int aggregate = WarpReduce(temp_storage[warp_id]).Sum(local_sum);

            if (cub::LaneId() == 0) {
                atomicAdd(p_total_out_degree, local_sum);
            }
        }

        template<typename TAppInst>
        __global__
        void CallPostComputation(TAppInst app_inst) {
            uint32_t tid = TID_1D;

            if (tid == 0) {
                app_inst.PostComputation();
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void SyncPullTD(TAppInst app_inst,
                        WorkSource work_source,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> node_in_buffer_datum,
                        GraphDatum<TBuffer> node_out_buffer_datum,
                        GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                sync_pull_td::Relax(app_inst,
                                    work_source,
                                    csc_graph,
                                    node_in_buffer_datum,
                                    node_out_buffer_datum,
                                    edge_weight_datum);
            } else {
                sync_pull_td::RelaxCTA<LB>(app_inst,
                                           work_source,
                                           csc_graph,
                                           node_in_buffer_datum,
                                           node_out_buffer_datum,
                                           edge_weight_datum);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename CSRGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushTD(TAppInst app_inst,
                         WorkSource work_source,
                         CSRGraph csr_graph,
                         GraphDatum<TValue> node_value_datum,
                         GraphDatum<TBuffer> node_buffer_datum,
                         GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                async_push_td::Relax<false>(app_inst,
                                            work_source,
                                            csr_graph,
                                            node_value_datum,
                                            node_buffer_datum,
                                            edge_weight_datum,
                                            (TBuffer) 0);
            } else {
                async_push_td::RelaxCTA<LB, false>(app_inst,
                                                   work_source,
                                                   csr_graph,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                   edge_weight_datum,
                                                   (TBuffer) 0);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename CSRGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushTDPrio(TAppInst app_inst,
                             WorkSource work_source,
                             CSRGraph csr_graph,
                             GraphDatum<TValue> node_value_datum,
                             GraphDatum<TBuffer> node_buffer_datum,
                             GraphDatum<TWeight> edge_weight_datum,
                             TBuffer current_priority) {
            if (LB == LoadBalancing::NONE) {
                async_push_td::Relax<true>(app_inst,
                                           work_source,
                                           csr_graph,
                                           node_value_datum,
                                           node_buffer_datum,
                                           edge_weight_datum,
                                           current_priority);
            } else {
                async_push_td::RelaxCTA<LB, true>(app_inst,
                                                  work_source,
                                                  csr_graph,
                                                  node_value_datum,
                                                  node_buffer_datum,
                                                  edge_weight_datum,
                                                  current_priority);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename CSRGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushTDFused(TAppInst app_inst,
                              WorkSource work_source,
                              CSRGraph csr_graph,
                              GraphDatum<TValue> node_value_datum,
                              GraphDatum<TBuffer> node_buffer_datum,
                              GraphDatum<TWeight> edge_weight_datum,
                              cub::GridBarrier grid_barrier,
                              uint32_t *p_active_count) {
            uint32_t work_size = work_source.get_size();
            uint32_t tid = TID_1D;

            while (*p_active_count) {
                if (LB == LoadBalancing::NONE) {
                    async_push_td::Relax<false>(app_inst,
                                                work_source,
                                                csr_graph,
                                                node_value_datum,
                                                node_buffer_datum,
                                                edge_weight_datum,
                                                (TBuffer) 0);
                } else {
                    async_push_td::RelaxCTA<LB, false>(app_inst,
                                                       work_source,
                                                       csr_graph,
                                                       node_value_datum,
                                                       node_buffer_datum,
                                                       edge_weight_datum,
                                                       (TBuffer) 0);
                }

                grid_barrier.Sync();

                if (tid == 0) {
//                        printf("Round: %u Policy to execute: ASYNC_PUSH_TD In: %u Out: %u",
//                               *app_inst.m_p_current_round, work_size, *p_active_count);
                    app_inst.PostComputation();
                    *p_active_count = 0;
                    *app_inst.m_p_current_round += 1;
                }
                grid_barrier.Sync();

                common::CountActiveNodes(app_inst,
                                         work_source,
                                         node_buffer_datum,
                                         p_active_count);
                grid_barrier.Sync();
            }

            // fix the last iteration times
            if (tid == 0) {
                *app_inst.m_p_current_round -= 1;
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename WorkTarget,
                typename CSRGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushDD(TAppInst app_inst,
                         WorkSource work_source,
                         WorkTarget work_target,
                         CSRGraph csr_graph,
                         GraphDatum<TValue> node_value_datum,
                         GraphDatum<TBuffer> node_buffer_datum,
                         GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                async_push_dd::Relax(app_inst,
                                     work_source,
                                     work_target,
                                     work_target,
                                     (TBuffer) 0,
                                     csr_graph,
                                     node_value_datum,
                                     node_buffer_datum,
                                     edge_weight_datum);
            } else {
                async_push_dd::RelaxCTA<LB>(app_inst,
                                            work_source,
                                            work_target,
                                            work_target,
                                            (TBuffer) 0,
                                            csr_graph,
                                            node_value_datum,
                                            node_buffer_datum,
                                            edge_weight_datum);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename WorkTarget,
                typename CSRGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushDDPrio(TAppInst app_inst,
                             WorkSource work_source,
                             WorkTarget work_target_low,
                             WorkTarget work_target_high,
                             TBuffer current_priority,
                             CSRGraph csr_graph,
                             GraphDatum<TValue> node_value_datum,
                             GraphDatum<TBuffer> node_buffer_datum,
                             GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                async_push_dd::Relax(app_inst,
                                     work_source,
                                     work_target_low,
                                     work_target_high,
                                     (TBuffer) current_priority,
                                     csr_graph,
                                     node_value_datum,
                                     node_buffer_datum,
                                     edge_weight_datum);
            } else {
                async_push_dd::RelaxCTA<LB>(app_inst,
                                            work_source,
                                            work_target_low,
                                            work_target_high,
                                            (TBuffer) current_priority,
                                            csr_graph,
                                            node_value_datum,
                                            node_buffer_datum,
                                            edge_weight_datum);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename CSRGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushDDFused(TAppInst app_inst,
                              groute::dev::Queue<index_t> queue_input,
                              groute::dev::Queue<index_t> queue_output,
                              CSRGraph csr_graph,
                              GraphDatum<TValue> node_value_datum,
                              GraphDatum<TBuffer> node_buffer_datum,
                              GraphDatum<TWeight> edge_weight_datum,
                              cub::GridBarrier grid_barrier) {
            uint32_t tid = TID_1D;
            groute::dev::Queue<index_t> *p_input = &queue_input;
            groute::dev::Queue<index_t> *p_output = &queue_output;

            assert(p_input->count() > 0);
            assert(p_output->count() == 0);

            while (p_input->count()) {
                groute::dev::WorkSourceArray<index_t> work_source(p_input->data_ptr(), p_input->count());

                auto &work_target = *p_output;

                if (LB == LoadBalancing::NONE) {
                    async_push_dd::Relax(app_inst,
                                         work_source,
                                         work_target,
                                         work_target,
                                         (TBuffer) 0,
                                         csr_graph,
                                         node_value_datum,
                                         node_buffer_datum,
                                         edge_weight_datum);
                } else {
                    async_push_dd::RelaxCTA<LB>(app_inst,
                                                work_source,
                                                work_target,
                                                work_target,
                                                (TBuffer) 0,
                                                csr_graph,
                                                node_value_datum,
                                                node_buffer_datum,
                                                edge_weight_datum);
                }

                grid_barrier.Sync(); // this barrier to ensure computation done

                if (tid == 0) {
//                        LOG("Round: %u In: %u Out: %u\n",
//                            *app_inst.m_p_current_round,
//                            work_source.get_size(),
//                            work_target.count());
                    app_inst.PostComputation();
                    *app_inst.m_p_current_round += 1;
                    p_input->reset();
                }

                utils::swap(p_input, p_output);
                grid_barrier.Sync(); // this barrier to ensure reset done
            }

            // fix the last iteration times
            if (tid == 0) {
                *app_inst.m_p_current_round -= 1;
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename CSRGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushDDFusedPrio(TAppInst app_inst,
                                  groute::dev::Queue<index_t> queue_input,
                                  groute::dev::Queue<index_t> queue_output_low,
                                  groute::dev::Queue<index_t> queue_output_high,
                                  TBuffer current_priority,
                                  CSRGraph csr_graph,
                                  GraphDatum<TValue> node_value_datum,
                                  GraphDatum<TBuffer> node_buffer_datum,
                                  GraphDatum<TWeight> edge_weight_datum,
                                  cub::GridBarrier grid_barrier) {
            const uint32_t tid = TID_1D;
            const TBuffer step = current_priority;
            groute::dev::Queue<index_t> *p_input = &queue_input;
            groute::dev::Queue<index_t> *p_output_low = &queue_output_low;
            groute::dev::Queue<index_t> *p_output_high = &queue_output_high;
            groute::dev::WorkSourceRange<index_t> work_source_all(0, csr_graph.nnodes);

            assert(p_input->count() > 0);
            assert(p_output_low->count() == 0);
            assert(p_output_high->count() == 0);

            while (p_input->count()) {
                while (p_input->count()) {
                    groute::dev::WorkSourceArray<index_t> work_source(p_input->data_ptr(), p_input->count());

                    if (LB == LoadBalancing::NONE) {
                        async_push_dd::Relax(app_inst,
                                             work_source,
                                             *p_output_low,
                                             *p_output_high,
                                             current_priority,
                                             csr_graph,
                                             node_value_datum,
                                             node_buffer_datum,
                                             edge_weight_datum);
                    } else {
                        async_push_dd::RelaxCTA<LB>(app_inst,
                                                    work_source,
                                                    *p_output_low,
                                                    *p_output_high,
                                                    current_priority,
                                                    csr_graph,
                                                    node_value_datum,
                                                    node_buffer_datum,
                                                    edge_weight_datum);
                    }
                    grid_barrier.Sync(); // this barrier to ensure computation done

                    if (tid == 0) {
//                            LOG("Round: %u In: %u Low: %u High: %u Prio: %u\n",
//                                *app_inst.m_p_current_round,
//                                work_source.get_size(),
//                                p_output_low->count(),
//                                p_output_high->count(),
//                                current_priority);
                        app_inst.PostComputation();
                        *app_inst.m_p_current_round += 1;
                        p_input->reset();
                    }
                    grid_barrier.Sync(); // wait for reset done
                    utils::swap(p_output_high, p_input);
                }
                current_priority += step;

                utils::swap(p_input, p_output_low);
                grid_barrier.Sync(); // this barrier to ensure reset done
            }

            // fix the last iteration times
            if (tid == 0) {
                *app_inst.m_p_current_round -= 1;
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename TBitmap,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPullDD(TAppInst app_inst,
                         WorkSource work_source,
                         TBitmap in_active,
                         TBitmap out_active,
                         CSCGraph csc_graph,
                         GraphDatum<TBuffer> node_buffer_datum,
                         GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                async_pull_dd::Relax(app_inst,
                                     work_source,
                                     in_active,
                                     out_active,
                                     out_active,
                                     (TBuffer) 0,
                                     csc_graph,
                                     node_buffer_datum,
                                     edge_weight_datum);
            } else {
                async_pull_dd::RelaxCTA<LB>(app_inst,
                                            work_source,
                                            in_active,
                                            out_active,
                                            out_active,
                                            (TBuffer) 0,
                                            csc_graph,
                                            node_buffer_datum,
                                            edge_weight_datum);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename TBitmap,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPullDDPrio(TAppInst app_inst,
                             WorkSource work_source,
                             TBitmap in_active,
                             TBitmap out_active_low,
                             TBitmap out_active_high,
                             TBuffer current_priority,
                             CSCGraph csc_graph,
                             GraphDatum<TBuffer> node_buffer_datum,
                             GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                async_pull_dd::Relax(app_inst,
                                     work_source,
                                     in_active,
                                     out_active_low,
                                     out_active_high,
                                     current_priority,
                                     csc_graph,
                                     node_buffer_datum,
                                     edge_weight_datum);
            } else {
                async_pull_dd::RelaxCTA<LB>(app_inst,
                                            work_source,
                                            in_active,
                                            out_active_low,
                                            out_active_high,
                                            current_priority,
                                            csc_graph,
                                            node_buffer_datum,
                                            edge_weight_datum);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename TBitmap,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void SyncPullDD(TAppInst app_inst,
                        WorkSource work_source,
                        TBitmap in_active,
                        TBitmap out_active,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> node_in_buffer_datum,
                        GraphDatum<TBuffer> node_out_buffer_datum,
                        GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                sync_pull_dd::Relax(app_inst,
                                    work_source,
                                    in_active,
                                    out_active,
                                    csc_graph,
                                    node_in_buffer_datum,
                                    node_out_buffer_datum,
                                    edge_weight_datum);
            } else {
                sync_pull_dd::RelaxCTA<LB>(app_inst,
                                           work_source,
                                           in_active,
                                           out_active,
                                           csc_graph,
                                           node_in_buffer_datum,
                                           node_out_buffer_datum,
                                           edge_weight_datum);
            }
        }
    }
}
#endif //HYBRID_DRIVER_H
