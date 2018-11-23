// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <gflags/gflags.h>
#include <utils/utils.h>
#include "gtools.h"

DEFINE_string(graphfile, "", "A file with a graph in Dimacs 10 format");
DEFINE_bool(ggr, true, "Graph file is a Galois binary GR file");
DEFINE_string(output, "", "edge list output");
DEFINE_string(out_degree, "", "Save out degree to file");
DEFINE_string(splitter, "\t", "the splitter between nodes pair");
DEFINE_bool(deself_cycle, false, "remove self-cycle, e.g. 0-0, 1-1 ...");
DEFINE_bool(deduplicate, true, "remove duplicated edges");
DEFINE_string(format, "gr", "graph format (gr/market/metis formats are supported)");
DEFINE_bool(d2ud, false, "convert directed graph to undirected graph");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    graph_t *graph;

    if (FLAGS_graphfile.empty()) {
        printf("A Graph File must be provided\n");
        exit(0);
    }

    printf("\nLoading graph %s (%d)\n", FLAGS_graphfile.substr(FLAGS_graphfile.find_last_of('\\') + 1).c_str(),
           FLAGS_ggr);
    graph = GetCachedGraph(FLAGS_graphfile, FLAGS_format);

    if (graph->nvtxs == 0) {
        printf("Empty graph!\n");
        exit(0);
    }

    idx_t nnodes = graph->nvtxs;
    idx_t nedges = graph->nedges;
    bool weighted_graph = graph->readew;
    idx_t *p_row_start = graph->xadj;
    idx_t *p_edge_dst = graph->adjncy;
    idx_t *p_weights = graph->adjwgt;

    if (weighted_graph) {
        printf("Weighted graph\n");
    } else {
        printf("Unweighted graph\n");
    }

    if (FLAGS_deself_cycle) {
        printf("Removing self-cylces\n");

        std::vector<idx_t> v_row_start(nnodes + 1, 0);
        std::vector<idx_t> v_edge_dst;
        std::vector<idx_t> v_edge_weight;

        nedges = 0;
        if (weighted_graph) {
            for (idx_t src = 0; src < nnodes; src++) {
                idx_t edge_begin = p_row_start[src];
                idx_t edge_end = p_row_start[src + 1];

                for (idx_t edge = edge_begin; edge < edge_end; edge++) {
                    idx_t dst = p_edge_dst[edge];
                    idx_t weight = p_weights[edge];

                    // remove self cycle
                    if (src != dst) {
                        v_edge_dst.push_back(dst);
                        v_edge_weight.push_back(weight);
                        nedges++;
                    }
                }

                v_row_start[src + 1] = nedges;
            }
        } else {
            for (idx_t src = 0; src < nnodes; src++) {
                idx_t edge_begin = p_row_start[src];
                idx_t edge_end = p_row_start[src + 1];

                for (idx_t edge = edge_begin; edge < edge_end; edge++) {
                    idx_t dst = p_edge_dst[edge];

                    // remove self cycle
                    if (src != dst) {
                        v_edge_dst.push_back(dst);
                        nedges++;
                    }
                }

                v_row_start[src + 1] = nedges;
            }
        }

        free(p_row_start);
        free(p_edge_dst);

        p_row_start = static_cast<idx_t *>(calloc(nnodes + 1, sizeof(idx_t)));
        if (p_row_start == nullptr) {
            printf("failure to allocate <p_row_start>\n");
            return 1;
        }

        p_edge_dst = static_cast<idx_t *>(calloc(nedges, sizeof(idx_t)));
        if (p_edge_dst == nullptr) {
            printf("failure to allocate <p_edge_dst>\n");
            return 1;
        }

        std::copy(v_row_start.begin(), v_row_start.end(), p_row_start);
        std::copy(v_edge_dst.begin(), v_edge_dst.end(), p_edge_dst);

        if (weighted_graph) {
            free(p_weights);
            p_weights = static_cast<idx_t *>(calloc(nedges, sizeof(idx_t)));

            if (p_weights == nullptr) {
                printf("failure to allocate <p_weights>\n");
                return 1;
            }

            std::copy(v_edge_weight.begin(), v_edge_weight.end(), p_weights);
        }

        printf("Self-cycle removed, edges: %d\n", nedges);
    }

    if (FLAGS_deduplicate) {
        printf("Removing duplicated edges\n");

        std::vector<idx_t> v_row_start(nnodes + 1, 0);
        std::vector<idx_t> v_edge_dst;
        std::vector<idx_t> v_edge_weight;
        nedges = 0;

        if (weighted_graph) {
            for (idx_t src = 0; src < nnodes; src++) {
                idx_t edge_begin = p_row_start[src];
                idx_t edge_end = p_row_start[src + 1];

                std::set<idx_t> set_dst;

                for (idx_t edge = edge_begin; edge < edge_end; edge++) {
                    idx_t dst = p_edge_dst[edge];
                    idx_t weight = p_weights[edge];

                    if (set_dst.find(dst) == set_dst.end()) {
                        set_dst.insert(dst);
                        v_edge_dst.push_back(dst);
                        v_edge_weight.push_back(weight);
                        nedges++;
                    }
                }

                v_row_start[src + 1] = nedges;
            }
        } else {
            for (idx_t src = 0; src < nnodes; src++) {
                idx_t edge_begin = p_row_start[src];
                idx_t edge_end = p_row_start[src + 1];

                std::set<idx_t> set_dst;

                for (idx_t edge = edge_begin; edge < edge_end; edge++) {
                    idx_t dst = p_edge_dst[edge];

                    if (set_dst.find(dst) == set_dst.end()) {
                        set_dst.insert(dst);
                        v_edge_dst.push_back(dst);
                        nedges++;
                    }
                }

                v_row_start[src + 1] = nedges;
            }
        }

        free(p_row_start);
        free(p_edge_dst);

        p_row_start = static_cast<idx_t *>(calloc(nnodes + 1, sizeof(idx_t)));
        if (p_row_start == nullptr) {
            printf("failure to allocate <p_row_start>\n");
            return 1;
        }

        p_edge_dst = static_cast<idx_t *>(calloc(nedges, sizeof(idx_t)));
        if (p_edge_dst == nullptr) {
            printf("failure to allocate <p_edge_dst>\n");
            return 1;
        }

        std::copy(v_row_start.begin(), v_row_start.end(), p_row_start);
        std::copy(v_edge_dst.begin(), v_edge_dst.end(), p_edge_dst);

        if (weighted_graph) {
            free(p_weights);
            p_weights = static_cast<idx_t *>(calloc(nedges, sizeof(idx_t)));

            if (p_weights == nullptr) {
                printf("failure to allocate <p_weights>\n");
                return 1;
            }

            std::copy(v_edge_weight.begin(), v_edge_weight.end(), p_weights);
        }

        printf("Deduplicated! edges: %u\n", nedges);
    }


    if (FLAGS_d2ud) {
        printf("Converting directed graph to undirected\n");

        if (weighted_graph) {
            std::vector<std::map<idx_t, idx_t >> srcVDst(nnodes);

            for (idx_t src = 0; src < nnodes; src++) {
                idx_t edge_begin = p_row_start[src];
                idx_t edge_end = p_row_start[src + 1];

                for (idx_t edge = edge_begin; edge < edge_end; edge++) {
                    idx_t dst = p_edge_dst[edge];
                    idx_t weight = p_weights[edge];

                    srcVDst[src].insert(std::pair<idx_t, idx_t>(dst, weight));
                    srcVDst[dst].insert(std::pair<idx_t, idx_t>(src, weight));
                }
            }

            free(p_row_start);
            free(p_edge_dst);
            free(p_weights);

            // re-calculate total edges after insert reversed edges
            nedges = 0;
            for (idx_t src = 0; src < nnodes; src++) {
                nedges += srcVDst[src].size();
            }

            p_row_start = static_cast<idx_t *>(calloc(nnodes + 1, sizeof(idx_t)));
            if (p_row_start == nullptr) {
                printf("failure to allocate <p_row_start>\n");
                return 1;
            }

            p_edge_dst = static_cast<idx_t *>(calloc(nedges, sizeof(idx_t)));
            if (p_edge_dst == nullptr) {
                printf("failure to allocate <p_edge_dst>\n");
                return 1;
            }

            p_weights = static_cast<idx_t *>(calloc(nedges, sizeof(idx_t)));
            if (p_weights == nullptr) {
                printf("failure to allocate <p_weights>\n");
                return 1;
            }

            nedges = 0;
            for (idx_t src = 0; src < nnodes; src++) {
                std::map<idx_t, idx_t> &VDstWeight = srcVDst[src];

                for (auto const &dstWeight:VDstWeight) {
                    p_edge_dst[nedges] = dstWeight.first;
                    p_weights[nedges] = dstWeight.second;
                    nedges++;
                }
                p_row_start[src + 1] = nedges;
            }
        } else {
            std::vector<std::set<idx_t>> srcVDst(nnodes);

            for (idx_t src = 0; src < nnodes; src++) {
                idx_t edge_begin = p_row_start[src];
                idx_t edge_end = p_row_start[src + 1];

                for (idx_t edge = edge_begin; edge < edge_end; edge++) {
                    idx_t dst = p_edge_dst[edge];

                    srcVDst[src].insert(dst);
                    srcVDst[dst].insert(src);
                }
            }

            nedges = 0;
            for (idx_t src = 0; src < nnodes; src++) {
                std::set<idx_t> &VDst = srcVDst[src];
                nedges += VDst.size();
            }


            free(p_row_start);
            free(p_edge_dst);

            printf("copy done\n");

            p_row_start = static_cast<idx_t *>(calloc(nnodes + 1, sizeof(idx_t)));
            if (p_row_start == nullptr) {
                printf("failure to allocate <p_row_start>\n");
                return 1;
            }

            p_edge_dst = static_cast<idx_t *>(calloc(nedges, sizeof(idx_t)));
            if (p_edge_dst == nullptr) {
                printf("failure to allocate <p_edge_dst>\n");
                return 1;
            }

            nedges = 0;
            for (idx_t src = 0; src < nnodes; src++) {
                std::set<idx_t> &VDst = srcVDst[src];

                for (idx_t dst:VDst) {
                    p_edge_dst[nedges++] = dst;
                }
                p_row_start[src + 1] = nedges;
            }
        }

        printf("Undirected graph generated, edges: %u\n", nedges);
    }

    if (!FLAGS_out_degree.empty()) {
        FILE *p_fo = fopen(FLAGS_out_degree.data(), "w");
        if (p_fo == nullptr) {
            fprintf(stderr, "can not open file %s", FLAGS_output.data());
            return 1;
        }
        for (idx_t src = 0; src < nnodes; src++) {
            idx_t edge_begin = p_row_start[src];
            idx_t edge_end = p_row_start[src + 1];
            idx_t out_degree = edge_end - edge_begin;

            fprintf(p_fo, "%u%s%u\n", src, FLAGS_splitter.data(), out_degree);
        }

        fclose(p_fo);

        printf("Out-degree written!\n");
    }

    if (!FLAGS_output.empty()) {
        FILE *p_fo = fopen(FLAGS_output.data(), "w");
        if (p_fo == nullptr) {
            fprintf(stderr, "can not open file %s", FLAGS_output.data());
            return 1;
        }

        printf("Writing to edgelist...\n");
        if (weighted_graph) {
            for (idx_t src = 0; src < nnodes; src++) {
                idx_t edge_begin = p_row_start[src];
                idx_t edge_end = p_row_start[src + 1];

                for (idx_t edge = edge_begin; edge < edge_end; edge++) {
                    idx_t dst = p_edge_dst[edge];
                    idx_t weight = p_weights[edge];

                    fprintf(p_fo, "%u%s%u%s%u\n", src, FLAGS_splitter.data(), dst, FLAGS_splitter.data(), weight);
                }
            }
        } else {
            for (idx_t src = 0; src < nnodes; src++) {
                idx_t edge_begin = p_row_start[src];
                idx_t edge_end = p_row_start[src + 1];

                for (idx_t edge = edge_begin; edge < edge_end; edge++) {
                    idx_t dst = p_edge_dst[edge];

                    fprintf(p_fo, "%u%s%u\n", src, FLAGS_splitter.data(), dst);
                }
            }
        }

        fclose(p_fo);

        printf("Edgelist written!\n");
    }

    free(p_row_start);
    free(p_edge_dst);
    if (weighted_graph) free(p_weights);


    return 0;
}