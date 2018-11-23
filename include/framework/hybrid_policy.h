// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_HYBRID_POLICY_H
#define HYBRID_HYBRID_POLICY_H

#include <set>
#include <string>
#include <algorithm>
#include <gflags/gflags.h>
#include <framework/common.h>

DECLARE_double(alpha);
DECLARE_double(beta);
DECLARE_bool(trace);
DECLARE_bool(estimate);

namespace sepgraph
{
    namespace policy
    {
        using common::AlgoVariant;
        using common::Scheduling;
        using common::Model;
        using common::MsgPassing;

        enum class AlgoType
        {
            ITERATIVE_SCHEME,
            TRAVERSAL_SCHEME,
        };

        typedef struct RunningInfo
        {
            AlgoType m_algo_type;
            AlgoVariant last_variant;

            // Graph related statistics
            uint32_t nnodes;
            uint32_t nedges;
            float power_law_alpha;
            float power_law_threshold;

            // Algorithm related statistics
            uint32_t input_active_count;
            uint32_t output_active_count;        // the number of nodes to be visited from the frontier
            uint32_t current_round;

            // Workload statistics
            uint32_t total_workload;           // edges to check
            uint32_t input_workload;
            uint32_t output_workload;

            std::map<AlgoVariant, float> policy_time;
            std::map<AlgoVariant, float> policy_predicted_time;

            // Running time statistics
            float time_load_graph;
            float time_init_graph;
            float time_total;
            float time_kernel;

            // Time for predicate
            float time_per_work; // only after this field when last round is DD

            // Time overhead
            float time_overhead_input_active_node;
            float time_overhead_output_active_node;
            float time_overhead_input_workload;
            float time_overhead_output_workload;
            float time_overhead_queue2bitmap;
            float time_overhead_bitmap2queue;
            float time_overhead_rebuild_worklist;
            float time_overhead_sample; // priority sample overhead
            float time_overhead_wl_sort; // sort worklist by node id
            float time_overhead_wl_unique; // deduplicate worklist

            RunningInfo(AlgoType algo_type) : m_algo_type(algo_type)
            {
                // Graph related statistics
                nnodes = 0;
                nedges = 0;
                power_law_alpha = 0; // TODO ,init
                power_law_threshold = 0;

                // Algorithm related statistics
                input_active_count = 0;
                output_active_count = 0;
                current_round = 0;

                // Workload statistics
                total_workload = 0;
                input_workload = 0;
                output_workload = 0;

                // Running time statistics
                time_load_graph = 0;
                time_init_graph = 0;
                time_total = 0;
                time_kernel = 0;

                // Time for predicate
                time_per_work = 0;

                // Time overhead
                time_overhead_input_active_node = 0;
                time_overhead_output_active_node = 0;
                time_overhead_input_workload = 0;
                time_overhead_output_workload = 0;
                time_overhead_rebuild_worklist = 0;
                time_overhead_queue2bitmap = 0;
                time_overhead_bitmap2queue = 0;
                time_overhead_sample = 0;
                time_overhead_wl_sort = 0;
                time_overhead_wl_unique = 0;
            }
        } TRunningInfo;


        class PolicyDecisionMaker
        {
        private:
            TRunningInfo &m_running_info;
            std::map<AlgoType, std::set<AlgoVariant>> m_opt_variants;
            std::vector<AlgoVariant> m_to_inspection;

            void PredicateRunningTime()
            {
                assert(m_running_info.m_algo_type == AlgoType::ITERATIVE_SCHEME);

                m_running_info.policy_predicted_time.clear();

                // Copy all Topology-Driven time
                for (const std::pair<AlgoVariant, float> &e : m_running_info.policy_time)
                {
                    if (e.first.m_scheduling == Scheduling::TOPOLOGY_DRIVEN)
                    {
                        m_running_info.policy_predicted_time[e.first] = m_running_info.policy_time[e.first];
                    }
                }

                // Calculate Data-driven time. Update for work/time
                if (m_running_info.last_variant.m_scheduling == Scheduling::DATA_DRIVEN)
                {
                    m_running_info.time_per_work =
                            m_running_info.policy_time[m_running_info.last_variant] / m_running_info.input_workload;
                }

                std::vector<std::pair<AlgoVariant, float>> dd_variants;

                std::copy_if(m_running_info.policy_time.begin(),
                             m_running_info.policy_time.end(),
                             std::back_inserter(dd_variants),
                             [](const std::pair<AlgoVariant, float> &pair)
                             {
                                 return pair.first.m_scheduling == Scheduling::DATA_DRIVEN;
                             });

                if (dd_variants.size() > 0)
                {
                    std::pair<AlgoVariant, float> &pair = dd_variants[0];
                    float time_data_driven = m_running_info.time_per_work * m_running_info.output_workload;

                    m_running_info.policy_predicted_time[pair.first] = time_data_driven;
                }
            }

        public:
            PolicyDecisionMaker(TRunningInfo &running_info) : m_running_info(running_info)
            {
                const AlgoType &algo_type = m_running_info.m_algo_type;
                /* ITERATIVE_SCHEME:
                 * SYNC_PULL_TD (Power-law)
                   ASYNC_PUSH_TD (normal case)
                   ASYNC_PUSH_DD (nearly convergence)
                */
                if (algo_type == AlgoType::ITERATIVE_SCHEME)
                {
                    m_opt_variants[algo_type].insert(AlgoVariant(
                            Model::SYNC, MsgPassing::PULL, Scheduling::TOPOLOGY_DRIVEN));

                    m_opt_variants[algo_type].insert(AlgoVariant(
                            Model::ASYNC, MsgPassing::PUSH, Scheduling::TOPOLOGY_DRIVEN));

                    m_opt_variants[algo_type].insert(AlgoVariant(
                            Model::ASYNC, MsgPassing::PUSH, Scheduling::DATA_DRIVEN));

                    for (const AlgoVariant &algo_var:m_opt_variants[algo_type])
                    {
                        m_to_inspection.push_back(algo_var);
                    }
                }
                else if (m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME)
                {
                    /* TRAVERSAL_SCHEME
                     * ASYNC_PUSH_DD scout_count(out-degree from input-worklist) <= edges_to_check / alpha
                     * SYNC_PULL_DD  scout_count > edges_to_check / alpha
                     * Continue SYNC_PULL_DD if awake_count >= old_awake_count or awake_count > nnodes / beta
                     */

                    AlgoVariant push(
                            Model::ASYNC, MsgPassing::PUSH, Scheduling::DATA_DRIVEN);
                    AlgoVariant pull(
                            Model::SYNC, MsgPassing::PULL, Scheduling::DATA_DRIVEN);

                    m_opt_variants[algo_type].insert(push);
                    m_opt_variants[algo_type].insert(pull);
                }
            }

            /**
             *
             * @return return a pair which means the policy to be executed and how many times needs to execute
             */
            AlgoVariant GetInitPolicy()
            {
                AlgoVariant init_policy;

                if (m_running_info.m_algo_type == AlgoType::ITERATIVE_SCHEME)
                {
                    const AlgoVariant algo_variant = *m_to_inspection.begin();
                    m_to_inspection.erase(m_to_inspection.begin());

                    init_policy = algo_variant;
                }
                else if (m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME)
                {
                    init_policy = AlgoVariant::ASYNC_PUSH_DD;
                }

                return init_policy;
            }

            AlgoVariant GetNextPolicy()
            {
                AlgoVariant next_policy;
                double ALPHA = FLAGS_alpha, BETA = FLAGS_beta;

                if (m_running_info.m_algo_type == AlgoType::ITERATIVE_SCHEME)
                {
                    // If we still have variants to insepect, we get N remove one
                    if (m_to_inspection.size() > 0)
                    {
                        const AlgoVariant algo_var = *m_to_inspection.begin();

                        m_to_inspection.erase(m_to_inspection.begin());
                        next_policy = algo_var;
                    }
                    else
                    {
                        PredicateRunningTime();
                        const AlgoVariant &algo_var = std::min_element(m_running_info.policy_predicted_time.begin(),
                                                                       m_running_info.policy_predicted_time.end(),
                                                                       [](const std::pair<AlgoVariant, float> &lhs,
                                                                          const std::pair<AlgoVariant, float> &rhs)
                                                                       {
                                                                           return lhs.second < rhs.second;
                                                                       })->first;
                        next_policy = algo_var;
                    }
                }
                else if (m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME)
                {
                    if (m_running_info.last_variant.m_msg_passing == MsgPassing::PULL)
                    {
                        int increased_nodes = m_running_info.output_active_count - m_running_info.input_active_count;
                        //increased_nodes > 0 ||
                        if ( m_running_info.output_active_count > m_running_info.nnodes / BETA)
                        {
                            next_policy = AlgoVariant::ASYNC_PULL_DD;
                            return next_policy;
                        }
                    }

                    assert(ALPHA > 0);
                    assert(BETA > 0);
                    if (m_running_info.output_workload > m_running_info.total_workload / ALPHA)
                    {
                        next_policy = AlgoVariant::ASYNC_PULL_DD;
                    }
                    else
                    {
                        next_policy = AlgoVariant::ASYNC_PUSH_DD;
                    }
                }

                return next_policy;
            }

            void PrintInfo()
            {
                printf("------------ Running time ------------\n");
                for (std::pair<AlgoVariant, float> e:m_running_info.policy_time)
                {
                    printf("%s %f\n", e.first.ToString().data(), e.second);
                }

                if (m_running_info.policy_predicted_time.size() > 0)
                {
                    printf("------------ Predicated time ------------\n");
                    for (std::pair<AlgoVariant, float> e:m_running_info.policy_predicted_time)
                    {
                        printf("%s %f\n", e.first.ToString().data(), e.second);
                    }
                }
            }

        };
    }
}
#endif //HYBRID_HYBRID_POLICY_H
