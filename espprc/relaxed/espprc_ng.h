#pragma once
// ============================================================================
// ESPPRC ng-route Relaxation — Label-Setting with Neighborhood Memory
// ============================================================================
// Relaxes elementarity: each node j only "remembers" visited nodes within
// its Δ-nearest neighborhood N(j). Nodes outside N(j) are forgotten,
// allowing revisits through distant detours.
//
// State transition:  new_ng = (old_ng ∩ N(j)) ∪ {j}
// vs exact:          new_visited = old_visited ∪ {j}
//
// State space: O(n · 2^Δ) vs O(n · 2^n) for exact.
// Δ=n → equivalent to exact ESPPRC.
//
// The optimal ng-RC is a valid LOWER BOUND on exact ESPPRC optimal RC.
// ============================================================================

#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <numeric>

namespace espprc_ng {

struct Result {
    double optimal_rc;
    int    optimal_cost;
    std::vector<int> optimal_path;
    bool   path_is_elementary;      // true if best path has no repeated nodes
    int    neg_rc_count;
    int    labels_created;
    double time_ms;
    bool   timed_out;
    int    delta;                   // neighborhood size used
};

// Precompute neighborhood masks: for each customer j, the Δ nearest customers
inline std::vector<uint64_t> build_ng_masks(
    const std::vector<std::vector<int>>& dist,
    int n_cust, int delta)
{
    int N = n_cust + 1;
    // ng_mask[j] = bitmask of j's Δ nearest customers (bit k-1 = customer k)
    // ng_mask[0] = 0 (depot has no neighborhood constraint)
    std::vector<uint64_t> ng_mask(N, 0u);

    for (int j = 1; j <= n_cust; j++) {
        // Sort other customers by distance to j
        std::vector<int> others(n_cust);
        std::iota(others.begin(), others.end(), 1);  // [1, 2, ..., n_cust]

        std::sort(others.begin(), others.end(), [&](int a, int b) {
            return dist[j][a] < dist[j][b];
        });

        // Take Δ nearest (excluding j itself)
        uint64_t mask = (1ULL << (j - 1));  // always include j itself
        int count = 0;
        for (int k : others) {
            if (k == j) continue;
            mask |= (1ULL << (k - 1));
            if (++count >= delta) break;
        }
        ng_mask[j] = mask;
    }

    return ng_mask;
}

// Weighted neighborhood: hybrid distance + reduced cost selection.
// First half of delta slots: nearest by distance (geometric guarantee).
// Second half: nearest by rc(j,i) = dist(j,i) - max(π_i, 0) (risk-based).
// This ensures geometric coverage while protecting high-value nodes.
// When duals are zero, degrades to pure distance (same as build_ng_masks).
inline std::vector<uint64_t> build_ng_masks_weighted(
    const std::vector<std::vector<int>>& dist,
    int n_cust, int delta,
    const std::vector<double>& cover_duals)
{
    int N = n_cust + 1;
    std::vector<uint64_t> ng_mask(N, 0u);

    int base_slots = (delta + 1) / 2;  // ceil(delta/2) for distance
    int rc_slots   = delta - base_slots; // remaining for rc-weighted

    for (int j = 1; j <= n_cust; j++) {
        std::vector<int> others(n_cust);
        std::iota(others.begin(), others.end(), 1);

        // Phase 1: sort by distance, take base_slots nearest
        std::sort(others.begin(), others.end(), [&](int a, int b) {
            return dist[j][a] < dist[j][b];
        });

        uint64_t mask = (1ULL << (j - 1));  // always include j itself
        int count = 0;
        for (int k : others) {
            if (k == j) continue;
            mask |= (1ULL << (k - 1));
            if (++count >= base_slots) break;
        }

        // Phase 2: from remaining nodes, sort by rc, take rc_slots
        if (rc_slots > 0) {
            std::vector<int> remaining;
            for (int k = 1; k <= n_cust; k++) {
                if (k == j) continue;
                if (mask & (1ULL << (k - 1))) continue; // already selected
                remaining.push_back(k);
            }
            std::sort(remaining.begin(), remaining.end(), [&](int a, int b) {
                double rc_a = (double)dist[j][a] - std::max(cover_duals[a - 1], 0.0);
                double rc_b = (double)dist[j][b] - std::max(cover_duals[b - 1], 0.0);
                return rc_a < rc_b;
            });
            int added = 0;
            for (int k : remaining) {
                mask |= (1ULL << (k - 1));
                if (++added >= rc_slots) break;
            }
        }

        ng_mask[j] = mask;
    }

    return ng_mask;
}

inline Result solve(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    int delta = 8,
    double time_limit_ms = 60000.0)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n_cust = (int)cover_duals.size();
    int N = n_cust + 1;

    // Clamp delta to n_cust (delta=n_cust → exact)
    if (delta > n_cust) delta = n_cust;

    // Precompute neighborhood masks (weighted by duals)
    std::vector<uint64_t> ng_mask = build_ng_masks_weighted(dist, n_cust, delta, cover_duals);

    // Build modified cost matrix
    std::vector<double> c_bar(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c_bar[i * N + j] = (double)dist[i][j];
            if (j >= 1)
                c_bar[i * N + j] -= cover_duals[j - 1];
        }

    // Label structure (identical to exact, but vmask = ng_set)
    struct Label {
        double   rc;
        int      load;
        uint64_t vmask;   // ng_set: subset of N(current_node) ∪ {current_node}
        int      node;
        int      prev;
    };

    std::vector<Label> labels;
    labels.reserve(8192);
    std::vector<std::vector<int>> node_labels(N);

    labels.push_back({0.0, 0, 0u, 0, -1});
    node_labels[0].push_back(0);

    std::deque<int> queue;
    queue.push_back(0);

    double best_rc = 1e30;
    int    best_label = -1;
    int    neg_count = 0;

    // Dominance (identical to exact)
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

    // BFS
    bool timed_out = false;
    int check_count = 0;

    while (!queue.empty()) {
        int li = queue.front();
        queue.pop_front();

        if (++check_count >= 4096) {
            check_count = 0;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(now - t0).count();
            if (elapsed > time_limit_ms) { timed_out = true; break; }
        }

        bool active = false;
        for (int a : node_labels[labels[li].node])
            if (a == li) { active = true; break; }
        if (!active) continue;

        int      cur_node  = labels[li].node;
        double   cur_rc    = labels[li].rc;
        int      cur_load  = labels[li].load;
        uint64_t cur_vmask = labels[li].vmask;

        for (int j = 1; j <= n_cust; j++) {
            if (cur_vmask & (1ULL << (j - 1))) continue;  // j in ng_set → blocked

            int new_load = cur_load + demand[j];
            if (new_load > capacity) continue;

            double new_rc = cur_rc + c_bar[cur_node * N + j];

            // *** THE KEY DIFFERENCE: ng-route state transition ***
            // Forget nodes outside j's neighborhood, then add j
            uint64_t new_vmask = (cur_vmask & ng_mask[j]) | (1ULL << (j - 1));

            int new_li = add_label(j, new_rc, new_load, new_vmask, li);
            if (new_li >= 0)
                queue.push_back(new_li);
        }

        // Return to depot
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

    // Build result
    auto t1 = std::chrono::high_resolution_clock::now();

    Result res;
    res.labels_created = (int)labels.size();
    res.neg_rc_count   = neg_count;
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.timed_out = timed_out;
    res.delta = delta;

    if (best_label >= 0) {
        res.optimal_rc = best_rc;

        std::vector<int> path;
        for (int t = best_label; t >= 0; t = labels[t].prev)
            path.push_back(labels[t].node);
        std::reverse(path.begin(), path.end());
        path.push_back(0);
        res.optimal_path = std::move(path);

        // Check if path is elementary (no repeated customers)
        std::vector<bool> seen(n_cust + 1, false);
        res.path_is_elementary = true;
        for (int v : res.optimal_path) {
            if (v == 0) continue;
            if (seen[v]) { res.path_is_elementary = false; break; }
            seen[v] = true;
        }

        int cost = 0;
        for (int k = 0; k + 1 < (int)res.optimal_path.size(); k++)
            cost += dist[res.optimal_path[k]][res.optimal_path[k + 1]];
        res.optimal_cost = cost;
    } else {
        res.optimal_rc   = 1e30;
        res.optimal_cost = 0;
        res.path_is_elementary = true;
    }

    return res;
}

} // namespace espprc_ng
