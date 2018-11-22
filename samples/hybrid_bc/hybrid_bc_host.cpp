// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "hybrid_bc_common.h"
#include <utils/stopwatch.h>
#include <unordered_set>
#include <queue>

DEFINE_int32(src, 0, "The source node of BC algorithm");
DECLARE_bool(verbose);


std::pair<std::vector<centrality_t>, std::vector<sigma_t >> BetweennessCentralityHost(const groute::graphs::host::CSRGraph &graph,
                                                                                      index_t src)
{
    Stopwatch sw(true);

    std::vector<int> source_path(graph.nnodes, -1);
    std::vector<centrality_t> bc_values(graph.nnodes, 0.0);
    std::vector<sigma_t> sigmas(graph.nnodes, 0.0);

    source_path[src] = 0;
    int search_depth = 0;
    sigmas[src] = 1;

    std::queue<index_t> wl1, wl2;
    std::queue<index_t> *in_wl = &wl1, *out_wl = &wl2;

    in_wl->push(src);


    while (!in_wl->empty())
    {
        while (!in_wl->empty())
        {
            index_t node = in_wl->front();
            in_wl->pop();

            int nbr_dist = source_path[node] + 1;

            for (index_t edge = graph.begin_edge(node),
                         end_edge = graph.end_edge(node); edge < end_edge; edge++)
            {
                index_t dest = graph.edge_dest(edge);

                if (source_path[dest] == -1)
                {
                    source_path[dest] = nbr_dist;
                    sigmas[dest] += sigmas[node];

                    if (search_depth < nbr_dist)
                    {
                        search_depth = nbr_dist;
                    }

                    out_wl->push(dest);
                }
                else
                {
                    if (source_path[dest] == source_path[node] + 1)
                    {
                        sigmas[dest] += sigmas[node];
                    }
                }
            }
        }
        std::swap(in_wl, out_wl);
    }
    search_depth++;

    for (int iter = search_depth - 2; iter > 0; --iter)
    {
        for (index_t node = 0; node < graph.nnodes; node++)
        {
            if (source_path[node] == iter)
            {
                for (index_t edge = graph.begin_edge(node),
                             end_edge = graph.end_edge(node); edge < end_edge; edge++)
                {
                    index_t dest = graph.edge_dest(edge);

                    if (source_path[dest] == iter + 1)
                    {
                        bc_values[node] += 1.0f * sigmas[node] / sigmas[dest] *
                                           (1.0f + bc_values[dest]);
                    }
                }
            }
        }
    }

    for (index_t node = 0; node < graph.nnodes; node++)
    {
        bc_values[node] *= 0.5f;
    }
    sw.stop();

    if (FLAGS_verbose)
    {
        printf("\nBC Host: %f ms. \n", sw.ms());
    }

    return std::make_pair(bc_values, sigmas);
}

int BCCheckErrors(std::vector<float> &regression_bc_values, std::vector<float> &bc_values)
{
    if (bc_values.size() != regression_bc_values.size())
    {
        return std::abs((int64_t) bc_values.size() - (int64_t) regression_bc_values.size());
    }

    index_t nnodes = bc_values.size();
    int total_errors = 0;

    for (index_t node = 0; node < nnodes; node++)
    {
        if (fabs(bc_values[node] - regression_bc_values[node]) > 2)
        {
            fprintf(stderr, "node: %u bc regression: %f device: %f\n", node, regression_bc_values[node], bc_values[node]);
            total_errors++;
        }
    }

    printf("Total errors: %u\n", total_errors);

    return total_errors;
}

int BCOutput(const char *file, const std::vector<centrality_t> &bc_values)
{
    FILE *f;
    f = fopen(file, "w");

    if (f)
    {
        for (int i = 0; i < bc_values.size(); ++i)
        {
            fprintf(f, "%u %f\n", i, bc_values[i]);
        }
        fclose(f);

        return 1;
    }
    else
    {
        fprintf(stderr, "Could not open '%s' for writing\n", file);
        return 0;
    }
}