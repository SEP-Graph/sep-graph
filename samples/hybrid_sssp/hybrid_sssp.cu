// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <functional>
#include <map>
#include <framework/framework.cuh>
#include <framework/hybrid_policy.h>
#include <framework/clion_cuda.cuh>
#include <utils/cuda_utils.h>
#include "hybrid_sssp_common.h"
#include "../../include/groute/graphs/csr_graph.h"

DEFINE_int32(source_node,
             0, "The source node for the SSSP traversal (clamped to [0, nnodes-1])");
DEFINE_bool(sparse,
            false, "use async/push/dd + fusion for high-diameter");
DEFINE_string(variant,
              "", "force variant as async/sync push/pull dd/td");
DECLARE_int32(top_ranks);
DECLARE_bool(print_ranks);
DECLARE_string(output);
DECLARE_bool(check);
DECLARE_int32(prio_delta);

namespace hybrid_sssp
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct SSSP : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {
        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        index_t m_source_node;

        SSSP(index_t source_node) : m_source_node(source_node)
        {

        }

        __forceinline__ __device__

        TValue GetInitValue(index_t node) const override
        {
            return static_cast<TValue> (IDENTITY_ELEMENT);
        }

        __forceinline__ __device__

        TBuffer GetInitBuffer(index_t node) const override
        {
            TBuffer buffer;
            if (node == m_source_node)
            {
                buffer = 0;
            }
            else
            {
                buffer = IDENTITY_ELEMENT;
            }
            return buffer;
        }

        __forceinline__ __host__ __device__
        TBuffer GetIdentityElement() const override
        {
            return IDENTITY_ELEMENT;
        }

        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {

            TBuffer buffer = *p_buffer;
            bool schedule = false;

            if (this->m_msg_passing == sepgraph::common::MsgPassing::PUSH)
            {
                if (*p_value > buffer)
                {
                    *p_value = buffer;
                    schedule = true;
                }
            }
            else
            {
                schedule = true;

                if (*p_value > buffer)
                {
                    *p_value = buffer;
                }
            }
            return utils::pair<TBuffer, bool>(buffer, schedule);
        }

        __forceinline__ __device__
        int AccumulateBuffer(index_t src,
                             index_t dst,
                             TWeight weight,
                             TBuffer *p_buffer,
                             TBuffer buffer) override
        {
            int status;

            if (buffer != IDENTITY_ELEMENT)
            {
                if (buffer + weight < atomicMin(p_buffer, buffer + weight))
                {
                    status = this->ACCUMULATE_SUCCESS_CONTINUE;
                }
                else
                {
                    status = this->ACCUMULATE_FAILURE_CONTINUE;
                }
            }
            else
            {
                status = this->ACCUMULATE_FAILURE_CONTINUE;
            }

            return status;
        }

        __forceinline__ __device__

        bool IsActiveNode(index_t node, TBuffer buffer) const override
        {
            return buffer != IDENTITY_ELEMENT;
        }

        __forceinline__ __device__

        bool IsHighPriority(TBuffer current_priority, TBuffer buffer) const override
        {
            return current_priority > buffer;
        }
    };
}


/**
 * Δ = cw/d,
    where d is the average degree in the graph, w is the average
    edge weight, and c is the warp width (32 on our GPUs).
 * @return
 */

bool HybridSSSP()
{
    assert(UINT32_MAX == UINT_MAX);
    typedef sepgraph::engine::Engine<distance_t, distance_t, distance_t, hybrid_sssp::SSSP, index_t> HybridEngine;
    HybridEngine engine(sepgraph::policy::AlgoType::TRAVERSAL_SCHEME);
    engine.LoadGraph();

    index_t source_node = min(max((index_t) 0, (index_t) FLAGS_source_node), engine.GetGraphDatum().nnodes - 1);

    sepgraph::common::EngineOptions engine_opt;

    if (FLAGS_variant.size() > 0)
    {
        if (FLAGS_variant == "async_push_dd")
        {
            engine_opt.ForceVariant(sepgraph::common::AlgoVariant::ASYNC_PUSH_DD);
        }
        else if (FLAGS_variant == "async_pull_dd")
        {
            engine_opt.ForceVariant(sepgraph::common::AlgoVariant::ASYNC_PULL_DD);
        }
        else if (FLAGS_variant == "async_push_td")
        {
            engine_opt.ForceVariant(sepgraph::common::AlgoVariant::ASYNC_PUSH_TD);
        }
        else
        {
            fprintf(stderr, "unsupported variant\n");
            exit(1);
        }
    }

    groute::graphs::host::CSRGraph csr_graph = engine.CSRGraph();
    double weight_sum = 0;
    for (uint32_t edge = 0; edge < csr_graph.nedges; edge++)
    {
        weight_sum += csr_graph.edge_weights[edge];
    }

    /**
     * We select a similar heuristic, Δ = cw/d,
        where d is the average degree in the graph, w is the average
        edge weight, and c is the warp width (32 on our GPUs)
        Link: https://people.csail.mit.edu/jshun/papers/DBGO14.pdf
     */
    int init_prio = 32 * (weight_sum / csr_graph.nedges) /
                    (1.0 * csr_graph.nedges / csr_graph.nnodes);

    printf("Priority delta: %u\n", init_prio);

    if (FLAGS_sparse)
    {
        engine_opt.SetFused();
        engine_opt.SetTwoLevelBasedPriority(init_prio);
        engine_opt.ForceVariant(sepgraph::common::AlgoVariant::ASYNC_PUSH_DD);
        engine_opt.SetLoadBalancing(sepgraph::common::MsgPassing::PUSH, sepgraph::common::LoadBalancing::NONE);
    }

    if (FLAGS_prio_delta > 0)
    {
        printf("Enable priority for scale-free dataset\n");
        engine_opt.SetTwoLevelBasedPriority(FLAGS_prio_delta);
    }

    engine.SetOptions(engine_opt);
    engine.InitGraph(source_node);
    engine.Start();
    engine.PrintInfo();

    const auto &distances = engine.GatherValue();
    const auto *p_weight_datum =
            const_cast<sepgraph::graphs::GraphDatum<distance_t, distance_t, distance_t> &>(engine.GetGraphDatum()).m_csr_edge_weight_datum.GetHostDataPtr();

    if (FLAGS_check)
    {
        auto regression = SSSPHostNaive(engine.CSRGraph(), p_weight_datum, source_node);
        int errors = SSSPCheckErrors(distances, regression);
        printf("total errors: %d\n", errors);
    }
    else
    {
        printf("Warning: Result not checked\n");
    }

    if (FLAGS_output.length() > 0)
    {
        SSSPOutput(FLAGS_output.data(), distances);
    }
    return true;
}