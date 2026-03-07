#pragma once
// ============================================================================
// ESPPRC Solver — Label-Setting DP with Bitmask Visited Set
// ============================================================================
// Elementary Shortest Path Problem with Resource Constraints (CVRP pricing).
//
// Input:  graph + demands + capacity + dual values
// Output: minimum reduced-cost elementary route (depot → customers → depot)
//
// Edge weights: c_bar[i][j] = dist[i][j] - pi[j]  (can be negative)
// Dominance:    L1 dom L2 iff rc1<=rc2, load1<=load2, visited1 ⊆ visited2
//
// Complexity: O(n · 2^n) worst case (bitmask), practical for n <= ~25.
// ============================================================================

#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <chrono>

namespace espprc {

struct Result {
    double optimal_rc;              // minimum RC among all complete routes
    int    optimal_cost;            // actual distance of best route
    std::vector<int> optimal_path;  // 0 -> ... -> 0
    int    neg_rc_count;            // routes with RC < -1e-6
    int    labels_created;          // total labels generated
    double time_ms;                 // wall-clock solve time
    bool   timed_out;               // true if time limit reached
};

inline Result solve(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    double time_limit_ms = 60000.0)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n_cust = (int)cover_duals.size();
    int N = n_cust + 1;   // depot + customers

    // ── Build modified cost matrix (flat for cache efficiency) ──────────
    std::vector<double> c_bar(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c_bar[i * N + j] = (double)dist[i][j];
            if (j >= 1)
                c_bar[i * N + j] -= cover_duals[j - 1];
        }

    // ── Label structure ────────────────────────────────────────────────
    struct Label {
        double   rc;
        int      load;
        uint64_t vmask;   // bitmask of visited customers (bit j-1 = customer j)
        int      node;
        int      prev;    // parent label index (-1 for root)
    };

    std::vector<Label> labels;
    labels.reserve(8192);

    // Per-node list of active (non-dominated) label indices
    std::vector<std::vector<int>> node_labels(N);

    // Root label at depot
    labels.push_back({0.0, 0, 0u, 0, -1});
    node_labels[0].push_back(0);

    // BFS queue of label indices
    std::deque<int> queue;
    queue.push_back(0);

    // Track best complete route
    double best_rc = 1e30;
    int    best_label = -1;
    int    neg_count = 0;

    // ── Dominance functions ────────────────────────────────────────────
    auto is_dominated = [&](int node, double rc, int load, uint64_t vmask) {
        for (int li : node_labels[node]) {
            const Label& el = labels[li];
            if (el.rc <= rc + 1e-10 && el.load <= load
                && (el.vmask & vmask) == el.vmask)
                return true;
        }
        return false;
    };

    auto add_label = [&](int node, double rc, int load,
                         uint64_t vmask, int prev) -> int {
        if (is_dominated(node, rc, load, vmask))
            return -1;

        // Remove labels dominated by the new one
        auto& nl = node_labels[node];
        int w = 0;
        for (int k = 0; k < (int)nl.size(); k++) {
            const Label& el = labels[nl[k]];
            if (!(rc <= el.rc + 1e-10 && load <= el.load
                  && (vmask & el.vmask) == vmask))
                nl[w++] = nl[k];
        }
        nl.resize(w);

        int idx = (int)labels.size();
        labels.push_back({rc, load, vmask, node, prev});
        nl.push_back(idx);
        return idx;
    };

    // ── Main BFS loop ──────────────────────────────────────────────────
    bool timed_out = false;
    int check_count = 0;

    while (!queue.empty()) {
        int li = queue.front();
        queue.pop_front();

        // Periodic time check (every 4096 iterations)
        if (++check_count >= 4096) {
            check_count = 0;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(now - t0).count();
            if (elapsed > time_limit_ms) { timed_out = true; break; }
        }

        // Stale check: label may have been dominated after enqueue
        bool active = false;
        for (int a : node_labels[labels[li].node])
            if (a == li) { active = true; break; }
        if (!active) continue;

        int      cur_node  = labels[li].node;
        double   cur_rc    = labels[li].rc;
        int      cur_load  = labels[li].load;
        uint64_t cur_vmask = labels[li].vmask;

        // Extend to each unvisited customer
        for (int j = 1; j <= n_cust; j++) {
            if (cur_vmask & (1ULL << (j - 1))) continue;

            int new_load = cur_load + demand[j];
            if (new_load > capacity) continue;

            double   new_rc    = cur_rc + c_bar[cur_node * N + j];
            uint64_t new_vmask = cur_vmask | (1ULL << (j - 1));

            int new_li = add_label(j, new_rc, new_load, new_vmask, li);
            if (new_li >= 0)
                queue.push_back(new_li);
        }

        // Try completing: return to depot
        if (cur_node != 0) {
            double final_rc = cur_rc + c_bar[cur_node * N + 0];
            if (final_rc < best_rc) {
                best_rc = final_rc;
                best_label = li;
            }
            if (final_rc < -1e-6)
                neg_count++;
        }
    }

    // ── Build result ───────────────────────────────────────────────────
    auto t1 = std::chrono::high_resolution_clock::now();

    Result res;
    res.labels_created = (int)labels.size();
    res.neg_rc_count   = neg_count;
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.timed_out = timed_out;

    if (best_label >= 0) {
        res.optimal_rc = best_rc;

        // Reconstruct path: trace parent pointers
        std::vector<int> path;
        for (int t = best_label; t >= 0; t = labels[t].prev)
            path.push_back(labels[t].node);
        std::reverse(path.begin(), path.end());
        path.push_back(0);   // return to depot
        res.optimal_path = std::move(path);

        // Actual distance cost
        int cost = 0;
        for (int k = 0; k + 1 < (int)res.optimal_path.size(); k++)
            cost += dist[res.optimal_path[k]][res.optimal_path[k + 1]];
        res.optimal_cost = cost;
    } else {
        res.optimal_rc   = 1e30;
        res.optimal_cost = 0;
    }

    return res;
}

} // namespace espprc
