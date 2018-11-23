// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_GRAPH_DATUM_H
#define HYBRID_GRAPH_DATUM_H

#include <gflags/gflags.h>
#include <framework/common.h>
#include <framework/hybrid_policy.h>
#include <groute/device/bitmap_impls.h>
#include <groute/graphs/csr_graph.h>
#include <groute/device/queue.cuh>
#include <utils/cuda_utils.h>

#define PRIORITY_SAMPLE_SIZE 1000
DECLARE_double(wl_alloc_factor);

namespace sepgraph {
    namespace graphs {
        template<typename TValue,
                typename TBuffer,
                typename TWeight>
        struct GraphDatum {
            // Graph metadata
            uint32_t nnodes, nedges;

            // Worklist
            groute::Queue<index_t> m_wl_array_in; // Work-list in
            groute::Queue<index_t> m_wl_array_out_high; // Work-list out High priority
            groute::Queue<index_t> m_wl_array_out_low; // Work-list out Low priority
            groute::Queue<index_t> m_wl_middle;

            Bitmap m_wl_bitmap_in; // Work-list in
            Bitmap m_wl_bitmap_out_high; // Work-list out high
            Bitmap m_wl_bitmap_out_low; // Work-list out low
            Bitmap m_wl_bitmap_middle;

            utils::SharedValue<uint32_t> m_current_round;

            // In/Out-degree for every nodes
            utils::SharedArray<uint32_t> m_in_degree;
            utils::SharedArray<uint32_t> m_out_degree;

            // Total In/Out-degree
            utils::SharedValue<uint32_t> m_total_in_degree;
            utils::SharedValue<uint32_t> m_total_out_degree;

            // Graph data
            groute::graphs::single::NodeOutputDatum<TValue> m_node_value_datum;
            groute::graphs::single::NodeOutputDatum<TBuffer> m_node_buffer_datum;
            groute::graphs::single::NodeOutputDatum<TBuffer> m_node_buffer_tmp_datum; // For sync algorithms
            groute::graphs::single::EdgeInputDatum<TWeight> m_csr_edge_weight_datum;
            groute::graphs::single::EdgeInputDatum<TWeight> m_csc_edge_weight_datum;


            // Running data
            utils::SharedValue<uint32_t> m_active_nodes;

            // Sampling
            utils::SharedArray<index_t> m_sampled_nodes;
            utils::SharedArray<TBuffer> m_sampled_values;
            bool m_weighted;

            GraphDatum(const groute::graphs::host::CSRGraph &csr_graph,
                       const groute::graphs::host::CSCGraph &csc_graph) : nnodes(csc_graph.nnodes),
                                                                          nedges(csc_graph.nedges),
                                                                          m_in_degree(nullptr, 0),
                                                                          m_out_degree(nullptr, 0),
                                                                          m_sampled_nodes(nullptr, 0),
                                                                          m_sampled_values(nullptr, 0) {
                m_node_value_datum.Allocate(csr_graph);
                m_node_buffer_datum.Allocate(csr_graph);
                m_node_buffer_tmp_datum.Allocate(csr_graph);

                // Weighted graph
                if (typeid(TWeight) != typeid(groute::graphs::NoWeight)) {
                    m_csr_edge_weight_datum.Allocate(csr_graph);
                    m_csc_edge_weight_datum.Allocate(csc_graph);
                    m_weighted = true;
                }

                // Graph data
                m_in_degree = std::move(utils::SharedArray<uint32_t>(nnodes));
                m_out_degree = std::move(utils::SharedArray<uint32_t>(nnodes));

                // 2 type Worklist
                uint32_t capacity = nedges * FLAGS_wl_alloc_factor;
//                uint32_t capacity = nnodes;

                m_wl_array_in = std::move(groute::Queue<index_t>(capacity));
                m_wl_array_out_low = std::move(groute::Queue<index_t>(capacity));
                m_wl_array_out_high = std::move(groute::Queue<index_t>(capacity));
                m_wl_middle = std::move(groute::Queue<index_t>(nnodes)); // middle worklist for pull+dd

                m_wl_bitmap_in = std::move(Bitmap(nnodes));
                m_wl_bitmap_out_high = std::move(Bitmap(nnodes));
                m_wl_bitmap_out_low = std::move(Bitmap(nnodes));
                m_wl_bitmap_middle = std::move(Bitmap(nnodes));

                m_sampled_nodes = std::move(utils::SharedArray<index_t>(PRIORITY_SAMPLE_SIZE));
                m_sampled_values = std::move(utils::SharedArray<TBuffer>(PRIORITY_SAMPLE_SIZE));
            }

            GraphDatum(GraphDatum &&other) = delete;

            GraphDatum &operator=(const GraphDatum &other) = delete;

            GraphDatum &operator=(GraphDatum &&other) = delete;

            const groute::graphs::dev::GraphDatum<TValue> &GetValueDeviceObject() const {
                return m_node_value_datum.DeviceObject();
            }

            const groute::graphs::dev::GraphDatum<TBuffer> &GetBufferDeviceObject() const {
                return m_node_buffer_datum.DeviceObject();
            }

            const groute::graphs::dev::GraphDatum<TBuffer> &GetBufferTmpDeviceObject() const {
                return m_node_buffer_tmp_datum.DeviceObject();
            }

            const groute::graphs::dev::GraphDatum<TWeight> &GetEdgeWeightDeviceObject() const {
                return m_csr_edge_weight_datum.DeviceObject();
            }

            const groute::graphs::dev::GraphDatum<TWeight> &GetCSCEdgeWeightDeviceObject() const {
                return m_csc_edge_weight_datum.DeviceObject();
            }

            const groute::dev::WorkSourceRange<index_t> GetWorkSourceRangeDeviceObject() {
                return groute::dev::WorkSourceRange<index_t>(0, nnodes);
            }

            const std::vector<TValue> &GatherValue() {
                m_node_value_datum.Gather();
                return m_node_value_datum.GetHostData();
            }

            const std::vector<TBuffer> &GatherBuffer() {
                m_node_buffer_datum.Gather();
                return m_node_buffer_datum.GetHostData();
            }
        };
    }
}

#endif //HYBRID_GRAPH_DATUM_H
