#pragma once
// ESPPRC Pricing Subproblem — Label-Setting with bitmask visited set.
// Standard dominance: L1 dom L2 iff cost1<=cost2, load1<=load2, S1⊆S2.
// Flat arrays, deque queue, index-based path reconstruction.

#include <vector>
#include <deque>
#include <cmath>
#include <cstdint>
#include <algorithm>

struct PricingColumn {
    int    cost;                // original route cost
    double rc;                  // reduced cost
    std::vector<int> visits;    // customer indices (1-based)
    std::vector<int> path;      // 0 → ... → 0
};

struct Label {
    double rc;
    int    load;
    uint32_t vmask;
    int    node;
    int    prev;    // parent label index (-1 for root)
};

inline std::vector<PricingColumn> solve_pricing(
    const int* dist, int n,                  // flat n*n distance matrix
    const int* demand, int capacity,
    const double* cover_duals, int n_customers,
    const double* c_bar,                     // pre-computed modified cost n*n
    const uint64_t* forbidden_mask,          // forbidden_mask[i]: bitmask of nodes forbidden from i
    int max_columns = 500)
{
    // Label storage (contiguous, reusable)
    std::vector<Label> all_labels;
    all_labels.reserve(4096);

    // Per-node list of active (non-dominated) label indices
    std::vector<std::vector<int>> node_labels(n);

    // Root label at depot
    all_labels.push_back({0.0, 0, 0u, 0, -1});
    node_labels[0].push_back(0);

    // BFS queue: label index
    std::deque<int> queue;
    queue.push_back(0);

    // Collected columns with negative RC
    std::vector<PricingColumn> columns;

    auto is_dominated = [&](int node, double rc, int load, uint32_t vmask) -> bool {
        for (int li : node_labels[node]) {
            const Label& el = all_labels[li];
            if (el.rc <= rc && el.load <= load && (el.vmask & vmask) == el.vmask)
                return true;
        }
        return false;
    };

    auto add_label = [&](int node, double rc, int load, uint32_t vmask, int prev) -> int {
        if (is_dominated(node, rc, load, vmask))
            return -1;

        // Remove labels dominated by new one
        auto& nl = node_labels[node];
        int write = 0;
        for (int k = 0; k < (int)nl.size(); k++) {
            const Label& el = all_labels[nl[k]];
            if (!(rc <= el.rc && load <= el.load && (vmask & el.vmask) == vmask))
                nl[write++] = nl[k];
        }
        nl.resize(write);

        int idx = (int)all_labels.size();
        all_labels.push_back({rc, load, vmask, node, prev});
        nl.push_back(idx);
        return idx;
    };

    while (!queue.empty()) {
        int li = queue.front();
        queue.pop_front();

        const Label& cur = all_labels[li];
        // Check if this label is still active (not dominated after enqueue)
        bool still_active = false;
        for (int active_li : node_labels[cur.node]) {
            if (active_li == li) { still_active = true; break; }
        }
        if (!still_active) continue;

        int cur_node = cur.node;
        double cur_rc = cur.rc;
        int cur_load = cur.load;
        uint32_t cur_vmask = cur.vmask;

        // Extend to each customer j
        for (int j = 1; j <= n_customers; j++) {
            if (cur_vmask & (1u << (j - 1))) continue;   // already visited
            if (forbidden_mask && (forbidden_mask[cur_node] & (1ull << j))) continue;

            int new_load = cur_load + demand[j];
            if (new_load > capacity) continue;

            double new_rc = cur_rc + c_bar[cur_node * n + j];
            uint32_t new_vmask = cur_vmask | (1u << (j - 1));

            int new_li = add_label(j, new_rc, new_load, new_vmask, li);
            if (new_li >= 0)
                queue.push_back(new_li);
        }

        // Try returning to depot
        if (cur_node != 0) {
            bool depot_forbidden = forbidden_mask && (forbidden_mask[cur_node] & 1ull);
            if (!depot_forbidden) {
                double final_rc = cur_rc + c_bar[cur_node * n + 0];
                if (final_rc < -1e-6) {
                    // Reconstruct path
                    std::vector<int> path;
                    int trace = li;
                    while (trace >= 0) {
                        path.push_back(all_labels[trace].node);
                        trace = all_labels[trace].prev;
                    }
                    std::reverse(path.begin(), path.end());
                    path.push_back(0);  // return to depot

                    // Compute original cost
                    int actual_cost = 0;
                    for (int k = 0; k + 1 < (int)path.size(); k++)
                        actual_cost += dist[path[k] * n + path[k+1]];

                    // Extract visits (non-depot nodes)
                    std::vector<int> visits;
                    for (int v : path)
                        if (v != 0) visits.push_back(v);

                    columns.push_back({actual_cost, final_rc, std::move(visits), std::move(path)});
                }
            }
        }
    }

    // Sort by reduced cost (most negative first), truncate
    std::sort(columns.begin(), columns.end(),
              [](const PricingColumn& a, const PricingColumn& b) { return a.rc < b.rc; });
    if ((int)columns.size() > max_columns)
        columns.resize(max_columns);

    return columns;
}

// Build modified cost matrix c_bar from duals and branching info
inline void build_modified_cost(
    double* c_bar, int n,
    const int* dist,
    const double* cover_duals,
    const double* edge_duals,           // flat n*n, 0 if no edge dual
    int n_customers)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c_bar[i * n + j] = (double)dist[i * n + j];
            if (j >= 1)
                c_bar[i * n + j] -= cover_duals[j - 1];
            if (edge_duals)
                c_bar[i * n + j] -= edge_duals[i * n + j];
        }
    }
}
