#pragma once
// ============================================================================
// ESPPRC Exact Ablation — runtime Pool / Bucket switches
// ============================================================================
// use_pool   : pre-allocate label pool (labels.reserve)
// use_bucket : load-bucket + RC-sort + min_rc skipping
//              if false: flat per-node list, linear scan, no sorting
//
// C0 = use_pool:false  use_bucket:false  (bare baseline)
// C1 = use_pool:true   use_bucket:false  (pool only)
// C2 = use_pool:false  use_bucket:true   (bucket only)
// C4 = use_pool:true   use_bucket:true   (== espprc_exact_pool)
// ============================================================================

#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <chrono>

namespace espprc_exact_ablation {

struct Result {
    double optimal_rc;
    int    optimal_cost;
    std::vector<int> optimal_path;
    bool   path_is_elementary;
    int    neg_rc_count;
    int    labels_created;
    long long dom_checks;
    double time_ms;
    bool   timed_out;
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
    bool use_pool,
    bool use_bucket,
    int theta = 20,
    double time_limit_ms = 60000.0)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n_cust = (int)cover_duals.size();
    int N = n_cust + 1;

    // Flat cost matrix
    std::vector<double> c_bar(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c_bar[i * N + j] = (double)dist[i][j];
            if (j >= 1)
                c_bar[i * N + j] -= cover_duals[j - 1];
        }

    // Label pool
    int est_labels = std::min(2000000, n_cust * (1 << std::min(n_cust, 16)));
    est_labels = std::max(est_labels, 8192);

    std::vector<Label> labels;
    if (use_pool)
        labels.reserve(est_labels);

    // ── Bucket structures (use_bucket=true) ──────────────────────────────────
    int step     = std::max(1, capacity / theta);
    int n_buckets = capacity / step + 1;

    // node_buckets[v][b] = label indices at node v in load bucket b, sorted by RC
    std::vector<std::vector<std::vector<int>>> node_buckets;
    // min_rc[v][b] = minimum RC in that bucket (for fast skip)
    std::vector<std::vector<double>> min_rc_bucket;

    // ── Flat structures (use_bucket=false) ───────────────────────────────────
    // node_flat[v] = all active label indices at node v, unsorted
    std::vector<std::vector<int>> node_flat;

    if (use_bucket) {
        node_buckets.assign(N, std::vector<std::vector<int>>(n_buckets));
        min_rc_bucket.assign(N, std::vector<double>(n_buckets, 1e30));
        int avg = std::max(4, est_labels / (N * n_buckets));
        for (int v = 0; v < N; v++)
            for (int b = 0; b < n_buckets; b++)
                node_buckets[v][b].reserve(avg);
    } else {
        node_flat.assign(N, std::vector<int>());
    }

    long long dom_checks = 0;

    // ── is_dominated ─────────────────────────────────────────────────────────
    auto is_dominated_bucket = [&](int node, double rc, int load, uint64_t vmask) {
        const Label* ldata = labels.data();
        int b_max = std::min(load / step, n_buckets - 1);
        for (int b = 0; b <= b_max; b++) {
            if (min_rc_bucket[node][b] > rc + 1e-10) continue;
            const auto& bl = node_buckets[node][b];
            for (int idx : bl) {
                const Label& el = ldata[idx];
                if (el.rc > rc + 1e-10) break; // RC-sorted: early stop
                dom_checks++;
                if (el.load <= load && (el.vmask & vmask) == el.vmask)
                    return true;
            }
        }
        return false;
    };

    auto is_dominated_flat = [&](int node, double rc, int load, uint64_t vmask) {
        const Label* ldata = labels.data();
        for (int idx : node_flat[node]) {
            const Label& el = ldata[idx];
            dom_checks++;
            if (el.rc <= rc + 1e-10 && el.load <= load && (el.vmask & vmask) == el.vmask)
                return true;
        }
        return false;
    };

    // ── add_label ─────────────────────────────────────────────────────────────
    auto add_label_bucket = [&](int node, double rc, int load,
                                uint64_t vmask, int prev) -> int {
        if (is_dominated_bucket(node, rc, load, vmask)) return -1;

        int b_new = std::min(load / step, n_buckets - 1);

        // Forward-domination: remove labels that are now dominated by the new one
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
                for (int li : bl) if (labels[li].rc < mr) mr = labels[li].rc;
                min_rc_bucket[node][b] = mr;
            }
        }

        int idx = (int)labels.size();
        labels.push_back({rc, load, vmask, node, prev, true});

        auto& bl = node_buckets[node][b_new];
        auto pos = std::lower_bound(bl.begin(), bl.end(), idx,
            [&](int a, int) { return labels[a].rc < rc; });
        bl.insert(pos, idx);

        if (rc < min_rc_bucket[node][b_new])
            min_rc_bucket[node][b_new] = rc;

        return idx;
    };

    auto add_label_flat = [&](int node, double rc, int load,
                              uint64_t vmask, int prev) -> int {
        if (is_dominated_flat(node, rc, load, vmask)) return -1;

        // Forward-domination: remove labels dominated by new one
        auto& fl = node_flat[node];
        int w = 0;
        for (int k = 0; k < (int)fl.size(); k++) {
            Label& el = labels[fl[k]];
            if (rc <= el.rc + 1e-10 && load <= el.load
                && (vmask & el.vmask) == vmask) {
                el.active = false;
            } else {
                fl[w++] = fl[k];
            }
        }
        fl.resize(w);

        int idx = (int)labels.size();
        labels.push_back({rc, load, vmask, node, prev, true});
        fl.push_back(idx);
        return idx;
    };

    // ── Seed ──────────────────────────────────────────────────────────────────
    labels.push_back({0.0, 0, 0u, 0, -1, true});
    if (use_bucket) {
        node_buckets[0][0].push_back(0);
        min_rc_bucket[0][0] = 0.0;
    } else {
        node_flat[0].push_back(0);
    }

    std::deque<int> queue;
    queue.push_back(0);

    double best_rc   = 1e30;
    int    best_label = -1;
    int    neg_count  = 0;
    bool   timed_out  = false;
    int    check_count = 0;

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

            double   new_rc    = cur_rc + cost_row[j];
            uint64_t new_vmask = cur_vmask | (1ULL << (j - 1));

            int new_li;
            if (use_bucket)
                new_li = add_label_bucket(j, new_rc, new_load, new_vmask, li);
            else
                new_li = add_label_flat(j, new_rc, new_load, new_vmask, li);

            if (new_li >= 0)
                queue.push_back(new_li);
        }

        if (cur_node != 0) {
            double final_rc = cur_rc + cost_row[0];
            if (final_rc < best_rc) {
                best_rc    = final_rc;
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
    res.neg_rc_count    = neg_count;
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.timed_out = timed_out;
    res.path_is_elementary = true;

    if (best_label >= 0) {
        res.optimal_rc = best_rc;

        std::vector<int> path;
        for (int t = best_label; t >= 0; t = labels[t].prev)
            path.push_back(labels[t].node);
        std::reverse(path.begin(), path.end());
        path.push_back(0);
        res.optimal_path = std::move(path);

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

} // namespace espprc_exact_ablation
