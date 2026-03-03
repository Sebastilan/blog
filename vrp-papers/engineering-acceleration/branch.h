#pragma once
// Edge Branching + Branch-and-Bound framework for BPC.

#include "types.h"
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <climits>

// A branching decision: forbid or force an edge
struct BranchDecision {
    int ek;             // edge_key(i, j, n) value
    bool is_force;      // true = force, false = forbid
};

// A B&B tree node
struct BBNode {
    std::vector<BranchDecision> decisions;
    double lb;
    int depth;
    int id;

    bool operator>(const BBNode& o) const { return lb > o.lb; }
};

inline int next_node_id() {
    static int counter = 0;
    return ++counter;
}

inline void reset_node_counter() {
    // Reset by calling next_node_id enough times...
    // Actually use a global. We'll manage this differently.
}

// Extract forbidden and forced edge key sets from a node's decisions
inline void get_branch_sets(const BBNode& node, int n,
                            std::set<int>& forbidden, std::set<int>& forced) {
    forbidden.clear();
    forced.clear();
    for (auto& d : node.decisions) {
        if (d.is_force)
            forced.insert(d.ek);
        else
            forbidden.insert(d.ek);
    }
}

// Build forbidden_mask array from forbidden edge set
// forbidden_mask[i] has bit j set if edge (i,j) is forbidden
inline void build_forbidden_mask(const std::set<int>& forbidden_keys, int n,
                                  std::vector<uint64_t>& mask) {
    mask.assign(n, 0ull);
    for (int ek : forbidden_keys) {
        int i = ek / n;
        int j = ek % n;
        mask[i] |= (1ull << j);
        mask[j] |= (1ull << i);
    }
}

// Build edge_duals flat array from forced edge dual map
inline void build_edge_duals_flat(const std::map<int, double>& edge_duals, int n,
                                   std::vector<double>& flat) {
    flat.assign(n * n, 0.0);
    for (auto& [ek, dual] : edge_duals) {
        int i = ek / n;
        int j = ek % n;
        flat[i * n + j] += dual;
        flat[j * n + i] += dual;
    }
}

// Compute edge flows: x[e] = sum_r lambda_r * [route r uses edge e]
inline std::map<int, double> compute_edge_flows(
    const std::vector<double>& lambdas,
    const std::vector<RouteColumn>& columns, int n)
{
    std::map<int, double> flows;
    for (int r = 0; r < (int)columns.size(); r++) {
        if (lambdas[r] < 1e-8) continue;
        const auto& path = columns[r].path;
        for (int k = 0; k + 1 < (int)path.size(); k++) {
            int ek = edge_key(path[k], path[k+1], n);
            flows[ek] += lambdas[r];
        }
    }
    return flows;
}

// Select most fractional edge (closest to 0.5)
inline int select_branch_edge(const std::map<int, double>& flows) {
    int best_ek = -1;
    double best_score = -1.0;

    for (auto& [ek, flow] : flows) {
        double frac = flow - std::floor(flow);
        if (frac < 1e-6) continue;
        double score = 0.5 - std::abs(frac - 0.5);
        if (score > best_score) {
            best_score = score;
            best_ek = ek;
        }
    }
    return best_ek;
}

// Create two children: forbid (left) / force (right)
inline std::pair<BBNode, BBNode> create_children(
    const BBNode& parent, int branch_ek, double lb)
{
    BBNode left, right;
    left.decisions = parent.decisions;
    left.decisions.push_back({branch_ek, false /*forbid*/});
    left.lb = lb;
    left.depth = parent.depth + 1;
    left.id = next_node_id();

    right.decisions = parent.decisions;
    right.decisions.push_back({branch_ek, true /*force*/});
    right.lb = lb;
    right.depth = parent.depth + 1;
    right.id = next_node_id();

    return {left, right};
}

// Check if LP solution is integer
inline bool is_integer_solution(const std::vector<double>& lambdas, double tol = 1e-6) {
    for (double lam : lambdas) {
        if (std::abs(lam - std::round(lam)) > tol)
            return false;
    }
    return true;
}

// Extract active routes from integer solution
inline std::vector<std::pair<int, std::vector<int>>>
extract_solution_routes(const std::vector<double>& lambdas,
                        const std::vector<RouteColumn>& columns, double tol = 1e-6)
{
    std::vector<std::pair<int, std::vector<int>>> routes;
    for (int i = 0; i < (int)lambdas.size(); i++) {
        if (lambdas[i] > 1.0 - tol)
            routes.push_back({columns[i].cost, columns[i].path});
    }
    return routes;
}

// Nearest-neighbor heuristic for initial upper bound
inline std::pair<int, std::vector<std::pair<int, std::vector<int>>>>
greedy_upper_bound(const int* dist, const int* demand, int capacity,
                   int n, int n_customers)
{
    std::vector<bool> visited(n, false);
    visited[0] = true;
    int total_cost = 0;
    std::vector<std::pair<int, std::vector<int>>> routes;
    int remaining = n_customers;

    while (remaining > 0) {
        std::vector<int> route = {0};
        int load = 0;
        int current = 0;

        while (remaining > 0) {
            int best = -1;
            int best_d = INT32_MAX;
            for (int j = 1; j <= n_customers; j++) {
                if (!visited[j] && load + demand[j] <= capacity) {
                    int d = dist[current * n + j];
                    if (d < best_d) { best_d = d; best = j; }
                }
            }
            if (best < 0) break;
            route.push_back(best);
            load += demand[best];
            visited[best] = true;
            current = best;
            remaining--;
        }
        route.push_back(0);

        int cost = 0;
        for (int k = 0; k + 1 < (int)route.size(); k++)
            cost += dist[route[k] * n + route[k+1]];
        total_cost += cost;
        routes.push_back({cost, route});
    }
    return {total_cost, routes};
}
