#pragma once
// ============================================================================
// ESPPRC Exact (Full Bitmask) + Pool (RC-Sort + Bucket v2 + Pre-allocation)
// ============================================================================
// L3 exact pricing via full bitmask: guarantees elementary paths in one pass.
// Uses the same pool infrastructure as espprc_ng_pool, but replaces ng-mask
// with full bitmask.
//
// State transition (full elementary):
//   new_vmask = cur_vmask | {j}          (no forgetting, all visits tracked)
//   Extension blocked if: cur_vmask & {j} (already visited)
//
// Dominance: L1 dom L2 iff rc1<=rc2, load1<=load2, visited1⊆visited2 (stricter than ng-route)
//
// Limitation: n_customers <= 32 (uint64_t bitmask)
// ============================================================================

#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <chrono>

namespace espprc_exact_pool {

// max_reward[q] = max dual reward collectible within capacity q (0-1 knapsack).
// Prune if new_rc - max_reward[capacity - new_load] >= -1e-6.
inline std::vector<double> compute_max_reward(
    const std::vector<double>& cover_duals,
    const std::vector<int>& demand,
    int capacity)
{
    int n_cust = (int)cover_duals.size();
    std::vector<double> dp(capacity + 1, 0.0);
    for (int i = 0; i < n_cust; i++) {
        if (cover_duals[i] <= 0) continue;
        int d = demand[i + 1];
        for (int q = capacity; q >= d; q--) {
            double val = dp[q - d] + cover_duals[i];
            if (val > dp[q]) dp[q] = val;
        }
    }
    for (int q = 1; q <= capacity; q++)
        if (dp[q] < dp[q - 1]) dp[q] = dp[q - 1];
    return dp;
}

struct Result {
    double optimal_rc;
    int    optimal_cost;
    std::vector<int> optimal_path;
    bool   path_is_elementary;   // always true for this solver
    int    neg_rc_count;
    int    labels_created;
    long long dom_checks;
    long long dom_skipped;
    long long bucket_skipped;
    long long labels_pruned;     // pruned by completion bounds
    double time_ms;
    bool   timed_out;
    int    n_buckets;
};

struct Label {
    double   rc;
    int      load;
    uint64_t vmask;
    int      node;
    int      prev;
    bool     active;
};

inline Result solve(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    int theta = 20,
    double time_limit_ms = 60000.0,
    bool use_cb = false)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n_cust = (int)cover_duals.size();
    int N = n_cust + 1;

    // Completion bounds (0-1 knapsack)
    std::vector<double> max_reward;
    if (use_cb)
        max_reward = compute_max_reward(cover_duals, demand, capacity);

    // Flat cost matrix
    std::vector<double> c_bar(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c_bar[i * N + j] = (double)dist[i][j];
            if (j >= 1)
                c_bar[i * N + j] -= cover_duals[j - 1];
        }

    // Bucket parameters
    int step = std::max(1, capacity / theta);
    int n_buckets = capacity / step + 1;

    // Label pool estimate (exact is heavier than ng, but cap it)
    int est_labels = std::min(2000000, n_cust * (1 << std::min(n_cust, 16)));
    est_labels = std::max(est_labels, 8192);

    std::vector<Label> labels;
    labels.reserve(est_labels);

    std::vector<std::vector<std::vector<int>>> node_buckets(
        N, std::vector<std::vector<int>>(n_buckets));

    int avg_per_bucket = std::max(4, est_labels / (N * n_buckets));
    for (int v = 0; v < N; v++)
        for (int b = 0; b < n_buckets; b++)
            node_buckets[v][b].reserve(avg_per_bucket);

    std::vector<std::vector<double>> min_rc(
        N, std::vector<double>(n_buckets, 1e30));

    labels.push_back({0.0, 0, 0u, 0, -1, true});
    node_buckets[0][0].push_back(0);
    min_rc[0][0] = 0.0;

    std::deque<int> queue;
    queue.push_back(0);

    double best_rc = 1e30;
    int    best_label = -1;
    int    neg_count = 0;
    long long dom_checks = 0;
    long long dom_skipped = 0;
    long long bucket_skipped = 0;
    long long labels_pruned = 0;

    auto is_dominated = [&](int node, double rc, int load, uint64_t vmask) {
        const Label* ldata = labels.data();
        int b_max = std::min(load / step, n_buckets - 1);
        for (int b = 0; b <= b_max; b++) {
            if (min_rc[node][b] > rc + 1e-10) {
                bucket_skipped++;
                continue;
            }
            const auto& bl = node_buckets[node][b];
            const int* bdata = bl.data();
            const int bsize = (int)bl.size();
            for (int i = 0; i < bsize; i++) {
                const Label& el = ldata[bdata[i]];
                if (el.rc > rc + 1e-10) {
                    dom_skipped += bsize - i - 1;
                    break;
                }
                dom_checks++;
                if (el.load <= load && (el.vmask & vmask) == el.vmask)
                    return true;
            }
        }
        return false;
    };

    auto add_label = [&](int node, double rc, int load,
                         uint64_t vmask, int prev) -> int {
        if (is_dominated(node, rc, load, vmask))
            return -1;

        int b_new = std::min(load / step, n_buckets - 1);

        for (int b = b_new; b < n_buckets; b++) {
            auto& bl = node_buckets[node][b];
            if (bl.empty()) continue;

            int w = 0, k = 0;
            for (; k < (int)bl.size(); k++) {
                if (labels[bl[k]].rc >= rc - 1e-10) break;
                bl[w++] = bl[k];
            }
            bool removed_any = false;
            for (; k < (int)bl.size(); k++) {
                const Label& el = labels[bl[k]];
                if (rc <= el.rc + 1e-10 && load <= el.load
                    && (vmask & el.vmask) == vmask) {
                    labels[bl[k]].active = false;
                    removed_any = true;
                    continue;
                }
                bl[w++] = bl[k];
            }
            bl.resize(w);

            if (removed_any) {
                double mr = 1e30;
                for (int li : bl)
                    if (labels[li].rc < mr) mr = labels[li].rc;
                min_rc[node][b] = mr;
            }
        }

        int idx = (int)labels.size();
        labels.push_back({rc, load, vmask, node, prev, true});

        auto& bl = node_buckets[node][b_new];
        auto pos = std::lower_bound(bl.begin(), bl.end(), idx,
            [&](int a, int /*b*/) { return labels[a].rc < rc; });
        bl.insert(pos, idx);

        if (rc < min_rc[node][b_new])
            min_rc[node][b_new] = rc;

        return idx;
    };

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

        if (!labels[li].active) continue;

        int      cur_node  = labels[li].node;
        double   cur_rc    = labels[li].rc;
        int      cur_load  = labels[li].load;
        uint64_t cur_vmask = labels[li].vmask;

        const double* cost_row = &c_bar[cur_node * N];

        for (int j = 1; j <= n_cust; j++) {
            // Full elementary: skip if already visited
            if (cur_vmask & (1ULL << (j - 1))) continue;

            int new_load = cur_load + demand[j];
            if (new_load > capacity) continue;

            double new_rc = cur_rc + cost_row[j];

            // Completion bound pruning
            if (use_cb && new_rc - max_reward[capacity - new_load] >= -1e-6) {
                labels_pruned++;
                continue;
            }

            // *** Full bitmask: no forgetting, all visits tracked ***
            uint64_t new_vmask = cur_vmask | (1ULL << (j - 1));

            int new_li = add_label(j, new_rc, new_load, new_vmask, li);
            if (new_li >= 0)
                queue.push_back(new_li);
        }

        if (cur_node != 0) {
            double final_rc = cur_rc + cost_row[0];
            if (final_rc < best_rc) {
                best_rc = final_rc;
                best_label = li;
            }
            if (final_rc < -1e-6)
                neg_count++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    Result res;
    res.labels_created  = (int)labels.size();
    res.dom_checks      = dom_checks;
    res.dom_skipped     = dom_skipped;
    res.bucket_skipped  = bucket_skipped;
    res.labels_pruned   = labels_pruned;
    res.neg_rc_count    = neg_count;
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.timed_out = timed_out;
    res.n_buckets = n_buckets;

    if (best_label >= 0) {
        res.optimal_rc = best_rc;

        std::vector<int> path;
        for (int t = best_label; t >= 0; t = labels[t].prev)
            path.push_back(labels[t].node);
        std::reverse(path.begin(), path.end());
        path.push_back(0);
        res.optimal_path = std::move(path);
        res.path_is_elementary = true;  // guaranteed by full bitmask

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

} // namespace espprc_exact_pool
