// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_SYNC_PULL_TD_H
#define HYBRID_SYNC_PULL_TD_H

#include <groute/device/cta_scheduler_hybrid.cuh>
#include <framework/variants/pull_functor.h>
#include <framework/variants/common.cuh>

namespace sepgraph
{
    namespace kernel
    {
        namespace sync_pull_td
        {
            using sepgraph::common::LoadBalancing;

            template<typename TAppInst,
                    typename WorkSource,
                    typename CSCGraph,
                    template<typename> class GraphDatum,
                    typename TBuffer,
                    typename TWeight>
            __device__ __forceinline__
            void Relax(TAppInst app_inst,
                       WorkSource work_source,
                       CSCGraph csc_graph,
                       GraphDatum<TBuffer> node_in_buffer_datum,
                       GraphDatum<TBuffer> node_out_buffer_datum,
                       GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                PullFunctor<TAppInst, CSCGraph, GraphDatum, TBuffer, TWeight>
                        pull_functor(app_inst, csc_graph, node_in_buffer_datum, node_out_buffer_datum, edge_weight_datum);

                for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t dst = work_source.get_work(i);
                    index_t begin_edge = csc_graph.begin_edge(dst),
                            end_edge = csc_graph.end_edge(dst);

                    for (index_t edge = begin_edge; edge < end_edge; edge++)
                    {
                        if (!pull_functor(edge, dst))
                        {
                            break;
                        }
                    }
                }
            }

            template<LoadBalancing LB,
                    typename TAppInst,
                    typename WorkSource,
                    typename CSCGraph,
                    template<typename> class GraphDatum,
                    typename TBuffer,
                    typename TWeight>
            __device__ __forceinline__
            void RelaxCTA(TAppInst app_inst,
                          WorkSource work_source,
                          CSCGraph csc_graph,
                          GraphDatum<TBuffer> node_in_buffer_datum,
                          GraphDatum<TBuffer> node_out_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PullFunctor<TAppInst, CSCGraph, GraphDatum, TBuffer, TWeight>
                        pull_functor(app_inst, csc_graph, node_in_buffer_datum, node_out_buffer_datum, edge_weight_datum);


                for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<index_t> np_local = {0, 0};

                    if (i < work_size)
                    {
                        index_t dst = work_source.get_work(i);

                        np_local.start = csc_graph.begin_edge(dst);
                        np_local.size = csc_graph.end_edge(dst) - np_local.start;
                        np_local.meta_data = dst;
                    }

                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<index_t, groute::dev::LB_COARSE_GRAINED>::template
                            schedule(np_local, pull_functor);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<index_t, groute::dev::LB_FINE_GRAINED>::template
                            schedule(np_local, pull_functor);
                            break;
                        case LoadBalancing::HYBRID:
                            groute::dev::CTAWorkSchedulerNew<index_t, groute::dev::LB_HYBRID>::template
                            schedule(np_local, pull_functor);
                            break;
                        default:
                            assert(false);
                    }
                }
            }
        }
    }
}

#endif //HYBRID_SYNC_PULL_TD_H
