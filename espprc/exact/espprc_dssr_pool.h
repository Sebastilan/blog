#pragma once
// ============================================================================
// ESPPRC DSSR + Pool (RC-Sort + Bucket v2 + Pre-allocation)
// ============================================================================
// Exact L3 pricing via DSSR: iteratively tighten ng-neighborhoods until
// the best path is elementary.  Each pricing pass uses the full pool
// infrastructure (RC-sorted buckets, min_rc skip, active flag, pre-alloc).
//
// State transition (DSSR-enhanced):
//   new_vmask = (cur_vmask & (ng_mask[j] | critical_mask)) | {j}
//
// Outer loop:
//   1. Run optimized pricing pass with current critical_mask
//   2. Check best path for repeated nodes
//   3. Add repeated nodes to critical_mask
//   4. Repeat until elementary or no new critical nodes
// ============================================================================

#include "../relaxed/espprc_ng.h"   // for build_ng_masks()

#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <chrono>

namespace espprc_dssr_pool {

struct Result {
    double optimal_rc;
    int    optimal_cost;
    std::vector<int> optimal_path;
    bool   path_is_elementary;
    int    neg_rc_count;
    int    labels_created;      // total across all DSSR iterations
    long long dom_checks;
    long long dom_skipped;
    long long bucket_skipped;
    long long labels_pruned;    // pruned by completion bounds
    double time_ms;
    bool   timed_out;
    int    delta;
    int    n_buckets;
    int    dssr_iterations;
    int    critical_count;
};

struct Label {
    double   rc;
    int      load;
    uint64_t vmask;
    int      node;
    int      prev;
    bool     active;
};

// max_reward[q] = max dual reward within capacity q (0-1 knapsack)
inline std::vector<double> compute_max_reward_dssr(
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

inline Result solve(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    int delta = 8,
    int theta = 20,
    int max_dssr_iter = 20,
    double time_limit_ms = 60000.0,
    bool use_cb = false)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n_cust = (int)cover_duals.size();
    int N = n_cust + 1;

    if (delta > n_cust) delta = n_cust;

    std::vector<uint64_t> ng_mask = espprc_ng::build_ng_masks_weighted(dist, n_cust, delta, cover_duals);

    // Flat cost matrix (shared across DSSR iterations)
    std::vector<double> c_bar(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c_bar[i * N + j] = (double)dist[i][j];
            if (j >= 1)
                c_bar[i * N + j] -= cover_duals[j - 1];
        }

    // Bucket parameters (shared)
    int step = std::max(1, capacity / theta);
    int n_buckets = capacity / step + 1;

    // Label pool estimate
    int est_labels = std::min(500000, n_cust * (1 << std::min(delta, 14)));
    est_labels = std::max(est_labels, 8192);

    // Completion bounds
    std::vector<double> max_reward;
    if (use_cb)
        max_reward = compute_max_reward_dssr(cover_duals, demand, capacity);

    uint64_t critical_mask = 0;
    int total_labels = 0;
    long long total_dom_checks = 0;
    long long total_dom_skipped = 0;
    long long total_bucket_skipped = 0;
    long long total_labels_pruned = 0;
    int dssr_iter = 0;
    bool timed_out = false;
    int best_neg_count = 0;

    double best_rc = 1e30;
    std::vector<int> best_path;

    for (dssr_iter = 0; dssr_iter < max_dssr_iter && !timed_out; dssr_iter++) {

        // Time check
        {
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double, std::milli>(now - t0).count() > time_limit_ms) {
                timed_out = true;
                break;
            }
        }

        // ── One optimized pricing pass ───────────────────────────────────
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

        double pass_best_rc = 1e30;
        int    pass_best_label = -1;
        int    pass_neg_count = 0;
        long long dom_checks = 0;
        long long dom_skipped = 0;
        long long bucket_skipped = 0;
        long long labels_pruned = 0;
        int check_count = 0;

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
                if (cur_vmask & (1ULL << (j - 1))) continue;

                int new_load = cur_load + demand[j];
                if (new_load > capacity) continue;

                double new_rc = cur_rc + cost_row[j];

                // Completion bound pruning
                if (use_cb && new_rc - max_reward[capacity - new_load] >= -1e-6) {
                    labels_pruned++;
                    continue;
                }

                // *** DSSR-enhanced ng-route state transition ***
                uint64_t new_vmask = (cur_vmask & (ng_mask[j] | critical_mask))
                                   | (1ULL << (j - 1));

                int new_li = add_label(j, new_rc, new_load, new_vmask, li);
                if (new_li >= 0)
                    queue.push_back(new_li);
            }

            if (cur_node != 0) {
                double final_rc = cur_rc + cost_row[0];
                if (final_rc < pass_best_rc) {
                    pass_best_rc = final_rc;
                    pass_best_label = li;
                }
                if (final_rc < -1e-6)
                    pass_neg_count++;
            }
        }

        total_labels += (int)labels.size();
        total_dom_checks += dom_checks;
        total_dom_skipped += dom_skipped;
        total_bucket_skipped += bucket_skipped;
        total_labels_pruned += labels_pruned;

        // ── Reconstruct best path and check for cycles ───────────────
        if (pass_best_label < 0) break;

        std::vector<int> path;
        for (int t = pass_best_label; t >= 0; t = labels[t].prev)
            path.push_back(labels[t].node);
        std::reverse(path.begin(), path.end());
        path.push_back(0);

        best_rc = pass_best_rc;
        best_neg_count = pass_neg_count;
        best_path = path;

        // Find repeated nodes
        uint64_t new_critical = 0;
        std::vector<bool> seen(n_cust + 1, false);
        bool is_elementary = true;
        for (int v : path) {
            if (v == 0) continue;
            if (seen[v]) {
                is_elementary = false;
                new_critical |= (1ULL << (v - 1));
            }
            seen[v] = true;
        }

        if (is_elementary) break;

        uint64_t added = new_critical & ~critical_mask;
        if (added == 0) break;

        critical_mask |= new_critical;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // ── Build result ─────────────────────────────────────────────────
    Result res;
    res.labels_created  = total_labels;
    res.dom_checks      = total_dom_checks;
    res.dom_skipped     = total_dom_skipped;
    res.bucket_skipped  = total_bucket_skipped;
    res.labels_pruned   = total_labels_pruned;
    res.neg_rc_count    = best_neg_count;
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.timed_out       = timed_out;
    res.delta           = delta;
    res.n_buckets       = n_buckets;
    res.dssr_iterations = dssr_iter + 1;

    int cc = 0;
    for (int i = 0; i < n_cust; i++)
        if (critical_mask & (1ULL << i)) cc++;
    res.critical_count = cc;

    if (!best_path.empty()) {
        res.optimal_rc = best_rc;
        res.optimal_path = best_path;

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

} // namespace espprc_dssr_pool
