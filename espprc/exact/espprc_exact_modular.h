#pragma once
// ============================================================================
// ESPPRC Exact — Modular Ablation Solver
// ============================================================================
// Four runtime bool switches control optimization strategies independently:
//   use_pool    — pre-allocate label pool + reserve buckets
//   use_bucket  — load-bucketed dominance + RC-sort + min_rc skip
//   use_bidir   — bidirectional search (fwd load<=Q/2, bwd load<=Q-Q/2)
//   use_parallel— fwd/bwd in separate std::threads (only when use_bidir=true)
//
// C4 configuration (pool=ON, bucket=ON, bidir=OFF) replicates espprc_exact_pool.
// ============================================================================

#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

namespace espprc_exact_modular {

struct Result {
    double optimal_rc;
    int    optimal_cost;
    std::vector<int> optimal_path;
    bool   path_is_elementary;
    int    neg_rc_count;
    int    labels_created;      // total labels (fwd+bwd for bidir)
    long long dom_checks;
    long long dom_skipped;
    long long bucket_skipped;
    long long labels_pruned;    // pruned by completion bounds
    double time_ms;
    bool   timed_out;
};

// ─────────────────────────────────────────────────────────────────────────────
// Unidirectional forward label-setting (core logic, shared by multiple configs)
// ─────────────────────────────────────────────────────────────────────────────

struct Label {
    double   rc;
    int      load;
    uint64_t vmask;
    int      node;
    int      prev;
    bool     active;
};

// Runs a single forward or backward label-setting pass.
// direction=true  → forward  (arc cost = c_bar[cur][j])
// direction=false → backward (arc cost = c_bar[j][cur], reversed arcs)
// load_limit: stop extending when load > load_limit
struct PhaseResult {
    std::vector<Label> labels;
    // node_labels[v] = sorted list of label indices at v (sorted by rc)
    // For bucket mode: node_buckets[v][b]
    std::vector<std::vector<int>> node_flat;          // used when !use_bucket
    std::vector<std::vector<std::vector<int>>> node_buckets; // used when use_bucket
    std::vector<std::vector<double>> min_rc_bucket;
    long long dom_checks   = 0;
    long long dom_skipped  = 0;
    long long bucket_skipped = 0;
    long long labels_pruned  = 0;
    bool timed_out = false;
};

// max_reward[q] = max dual reward within capacity q (0-1 knapsack)
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

inline PhaseResult run_phase(
    bool direction,        // true=forward, false=backward
    int  load_limit,       // max load allowed for extension
    int  capacity,
    int  n_cust,
    const std::vector<double>& c_bar,
    const std::vector<int>&    demand,
    bool use_pool,
    bool use_bucket,
    bool use_cb,
    const std::vector<double>& max_reward,
    int  theta,
    double time_limit_ms,
    const std::chrono::time_point<std::chrono::high_resolution_clock>& t0)
{
    const int N = n_cust + 1;
    PhaseResult ph;

    // ── Pool / reserve setup ─────────────────────────────────────────
    int est_labels = use_pool
        ? std::min(2000000, n_cust * (1 << std::min(n_cust, 16)))
        : 4096;
    est_labels = std::max(est_labels, 8192);

    ph.labels.reserve(est_labels);

    // ── Bucket setup ─────────────────────────────────────────────────
    int step = 1, n_buckets = 1;
    if (use_bucket) {
        step = std::max(1, capacity / theta);
        n_buckets = capacity / step + 1;
    }

    if (use_bucket) {
        ph.node_buckets.assign(N, std::vector<std::vector<int>>(n_buckets));
        ph.min_rc_bucket.assign(N, std::vector<double>(n_buckets, 1e30));
        if (use_pool) {
            int avg = std::max(4, est_labels / (N * n_buckets));
            for (int v = 0; v < N; v++)
                for (int b = 0; b < n_buckets; b++)
                    ph.node_buckets[v][b].reserve(avg);
        }
    } else {
        ph.node_flat.resize(N);
    }

    // ── Root label ───────────────────────────────────────────────────
    ph.labels.push_back({0.0, 0, 0u, 0, -1, true});
    if (use_bucket) {
        ph.node_buckets[0][0].push_back(0);
        ph.min_rc_bucket[0][0] = 0.0;
    } else {
        ph.node_flat[0].push_back(0);
    }

    // ── Dominance check ─────────────────────────────────────────────
    auto is_dominated_bucket = [&](int node, double rc, int load, uint64_t vmask) -> bool {
        const Label* ldata = ph.labels.data();
        int b_max = std::min(load / step, n_buckets - 1);
        for (int b = 0; b <= b_max; b++) {
            if (ph.min_rc_bucket[node][b] > rc + 1e-10) {
                ph.bucket_skipped++;
                continue;
            }
            const auto& bl = ph.node_buckets[node][b];
            const int bsize = (int)bl.size();
            for (int i = 0; i < bsize; i++) {
                const Label& el = ldata[bl[i]];
                if (el.rc > rc + 1e-10) {
                    ph.dom_skipped += bsize - i - 1;
                    break;
                }
                ph.dom_checks++;
                if (el.load <= load && (el.vmask & vmask) == el.vmask)
                    return true;
            }
        }
        return false;
    };

    auto is_dominated_flat = [&](int node, double rc, int load, uint64_t vmask) -> bool {
        for (int li : ph.node_flat[node]) {
            const Label& el = ph.labels[li];
            ph.dom_checks++;
            if (el.rc <= rc + 1e-10 && el.load <= load && (el.vmask & vmask) == el.vmask)
                return true;
        }
        return false;
    };

    // ── Add label ────────────────────────────────────────────────────
    // Bucket mode: RC-sorted insertion + backward dominance pruning
    auto add_label_bucket = [&](int node, double rc, int load, uint64_t vmask, int prev) -> int {
        if (is_dominated_bucket(node, rc, load, vmask)) return -1;

        int b_new = std::min(load / step, n_buckets - 1);

        // Prune dominated labels in higher load buckets
        for (int b = b_new; b < n_buckets; b++) {
            auto& bl = ph.node_buckets[node][b];
            if (bl.empty()) continue;
            int w = 0, k = 0;
            // Keep labels with rc < new rc (they can't be dominated by new)
            for (; k < (int)bl.size(); k++) {
                if (ph.labels[bl[k]].rc >= rc - 1e-10) break;
                bl[w++] = bl[k];
            }
            bool removed_any = false;
            for (; k < (int)bl.size(); k++) {
                const Label& el = ph.labels[bl[k]];
                if (rc <= el.rc + 1e-10 && load <= el.load && (vmask & el.vmask) == vmask) {
                    ph.labels[bl[k]].active = false;
                    removed_any = true;
                    continue;
                }
                bl[w++] = bl[k];
            }
            bl.resize(w);
            if (removed_any) {
                double mr = 1e30;
                for (int li : bl) if (ph.labels[li].rc < mr) mr = ph.labels[li].rc;
                ph.min_rc_bucket[node][b] = mr;
            }
        }

        int idx = (int)ph.labels.size();
        ph.labels.push_back({rc, load, vmask, node, prev, true});

        auto& bl = ph.node_buckets[node][b_new];
        auto pos = std::lower_bound(bl.begin(), bl.end(), idx,
            [&](int a, int) { return ph.labels[a].rc < rc; });
        bl.insert(pos, idx);
        if (rc < ph.min_rc_bucket[node][b_new])
            ph.min_rc_bucket[node][b_new] = rc;

        return idx;
    };

    auto add_label_flat = [&](int node, double rc, int load, uint64_t vmask, int prev) -> int {
        if (is_dominated_flat(node, rc, load, vmask)) return -1;
        // Remove labels now dominated by new one
        auto& nl = ph.node_flat[node];
        int w = 0;
        for (int k = 0; k < (int)nl.size(); k++) {
            const Label& el = ph.labels[nl[k]];
            if (!(rc <= el.rc + 1e-10 && load <= el.load && (vmask & el.vmask) == vmask))
                nl[w++] = nl[k];
        }
        nl.resize(w);
        int idx = (int)ph.labels.size();
        ph.labels.push_back({rc, load, vmask, node, prev, true});
        nl.push_back(idx);
        return idx;
    };

    // ── BFS / label-setting loop ─────────────────────────────────────
    std::deque<int> queue;
    queue.push_back(0);
    int check_count = 0;

    while (!queue.empty()) {
        int li = queue.front();
        queue.pop_front();

        if (++check_count >= 4096) {
            check_count = 0;
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double, std::milli>(now - t0).count() > time_limit_ms) {
                ph.timed_out = true;
                break;
            }
        }

        // Staleness check
        if (use_bucket) {
            if (!ph.labels[li].active) continue;
        }

        int      cur_node  = ph.labels[li].node;
        double   cur_rc    = ph.labels[li].rc;
        int      cur_load  = ph.labels[li].load;
        uint64_t cur_vmask = ph.labels[li].vmask;

        // Bidir half-point: don't extend beyond load_limit
        if (cur_load > load_limit) continue;

        for (int j = 1; j <= n_cust; j++) {
            if (cur_vmask & (1ULL << (j - 1))) continue;

            int new_load = cur_load + demand[j];
            if (new_load > capacity) continue;

            double arc_cost = direction
                ? c_bar[cur_node * N + j]
                : c_bar[j * N + cur_node];
            double new_rc = cur_rc + arc_cost;

            // Completion bound pruning
            if (use_cb && new_rc - max_reward[capacity - new_load] >= -1e-6) {
                ph.labels_pruned++;
                continue;
            }

            uint64_t new_vmask = cur_vmask | (1ULL << (j - 1));

            int new_li = use_bucket
                ? add_label_bucket(j, new_rc, new_load, new_vmask, li)
                : add_label_flat  (j, new_rc, new_load, new_vmask, li);

            if (new_li >= 0)
                queue.push_back(new_li);
        }
    }

    return ph;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main solve entry point
// ─────────────────────────────────────────────────────────────────────────────
inline Result solve(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    bool use_pool,
    bool use_bucket,
    bool use_bidir,
    bool use_parallel,
    bool use_cb = false,
    int  theta = 20,
    double time_limit_ms = 60000.0)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n_cust = (int)cover_duals.size();
    int N = n_cust + 1;

    // Build flat c_bar matrix
    std::vector<double> c_bar(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c_bar[i * N + j] = (double)dist[i][j];
            if (j >= 1)
                c_bar[i * N + j] -= cover_duals[j - 1];
        }

    // Completion bounds (0-1 knapsack over all customers)
    std::vector<double> max_reward;
    if (use_cb)
        max_reward = compute_max_reward(cover_duals, demand, capacity);

    Result res;
    res.path_is_elementary = true;
    res.neg_rc_count = 0;
    res.labels_created = 0;
    res.dom_checks = 0;
    res.dom_skipped = 0;
    res.bucket_skipped = 0;
    res.labels_pruned = 0;
    res.timed_out = false;
    res.optimal_rc = 1e30;
    res.optimal_cost = 0;

    if (!use_bidir) {
        // ── Unidirectional forward-only ──────────────────────────────
        PhaseResult ph = run_phase(
            true, capacity, capacity, n_cust,
            c_bar, demand, use_pool, use_bucket, use_cb, max_reward,
            theta, time_limit_ms, t0);

        res.timed_out    = ph.timed_out;
        res.labels_created = (int)ph.labels.size();
        res.dom_checks   = ph.dom_checks;
        res.dom_skipped  = ph.dom_skipped;
        res.bucket_skipped = ph.bucket_skipped;
        res.labels_pruned  = ph.labels_pruned;

        // Find best complete route (return to depot)
        int best_label = -1;
        for (int li = 0; li < (int)ph.labels.size(); li++) {
            if (!ph.labels[li].active) continue;
            if (ph.labels[li].node == 0) continue;
            double final_rc = ph.labels[li].rc + c_bar[ph.labels[li].node * N + 0];
            if (final_rc < res.optimal_rc) {
                res.optimal_rc = final_rc;
                best_label = li;
            }
            if (final_rc < -1e-6) res.neg_rc_count++;
        }

        if (best_label >= 0) {
            std::vector<int> path;
            for (int t = best_label; t >= 0; t = ph.labels[t].prev)
                path.push_back(ph.labels[t].node);
            std::reverse(path.begin(), path.end());
            path.push_back(0);
            res.optimal_path = std::move(path);

            int cost = 0;
            for (int k = 0; k + 1 < (int)res.optimal_path.size(); k++)
                cost += dist[res.optimal_path[k]][res.optimal_path[k + 1]];
            res.optimal_cost = cost;
        }

    } else {
        // ── Bidirectional ─────────────────────────────────────────────
        int half = capacity / 2;
        int bwd_limit = capacity - half;

        PhaseResult fwd_ph, bwd_ph;

        if (use_parallel) {
            // Run fwd and bwd in parallel threads
            std::thread t_fwd([&]() {
                fwd_ph = run_phase(true,  half,      capacity, n_cust,
                                   c_bar, demand, use_pool, use_bucket,
                                   use_cb, max_reward, theta, time_limit_ms, t0);
            });
            std::thread t_bwd([&]() {
                bwd_ph = run_phase(false, bwd_limit, capacity, n_cust,
                                   c_bar, demand, use_pool, use_bucket,
                                   use_cb, max_reward, theta, time_limit_ms, t0);
            });
            t_fwd.join();
            t_bwd.join();
        } else {
            fwd_ph = run_phase(true,  half,      capacity, n_cust,
                               c_bar, demand, use_pool, use_bucket,
                               use_cb, max_reward, theta, time_limit_ms, t0);
            if (!fwd_ph.timed_out)
                bwd_ph = run_phase(false, bwd_limit, capacity, n_cust,
                                   c_bar, demand, use_pool, use_bucket,
                                   use_cb, max_reward, theta, time_limit_ms, t0);
        }

        res.timed_out = fwd_ph.timed_out || bwd_ph.timed_out;
        res.labels_created = (int)fwd_ph.labels.size() + (int)bwd_ph.labels.size();
        res.dom_checks   = fwd_ph.dom_checks   + bwd_ph.dom_checks;
        res.dom_skipped  = fwd_ph.dom_skipped  + bwd_ph.dom_skipped;
        res.bucket_skipped = fwd_ph.bucket_skipped + bwd_ph.bucket_skipped;
        res.labels_pruned  = fwd_ph.labels_pruned  + bwd_ph.labels_pruned;

        if (!res.timed_out) {
            // ── Merge phase ───────────────────────────────────────────
            // Collect active fwd/bwd labels per node
            std::vector<std::vector<int>> fwd_at(N), bwd_at(N);
            for (int li = 0; li < (int)fwd_ph.labels.size(); li++) {
                if (!fwd_ph.labels[li].active) continue;
                fwd_at[fwd_ph.labels[li].node].push_back(li);
            }
            for (int li = 0; li < (int)bwd_ph.labels.size(); li++) {
                if (!bwd_ph.labels[li].active) continue;
                bwd_at[bwd_ph.labels[li].node].push_back(li);
            }

            // Sort by rc for early stopping
            for (int v = 0; v < N; v++) {
                std::sort(fwd_at[v].begin(), fwd_at[v].end(),
                    [&](int a, int b) { return fwd_ph.labels[a].rc < fwd_ph.labels[b].rc; });
                std::sort(bwd_at[v].begin(), bwd_at[v].end(),
                    [&](int a, int b) { return bwd_ph.labels[a].rc < bwd_ph.labels[b].rc; });
            }

            int best_fi = -1, best_bi = -1;
            int merge_check = 0;

            for (int i = 0; i < N && !res.timed_out; i++) {
                if (fwd_at[i].empty()) continue;
                for (int j = 0; j < N && !res.timed_out; j++) {
                    if (i == j || bwd_at[j].empty()) continue;
                    double edge_cost = c_bar[i * N + j];

                    for (int fi : fwd_at[i]) {
                        const Label& fl = fwd_ph.labels[fi];
                        // Early stop: fl.rc + edge_cost + best_possible_bwd >= best_rc
                        if (fl.rc + edge_cost + bwd_ph.labels[bwd_at[j][0]].rc >= res.optimal_rc - 1e-10)
                            break;

                        for (int bi : bwd_at[j]) {
                            const Label& bl = bwd_ph.labels[bi];
                            // Early stop on bwd rc
                            if (fl.rc + edge_cost + bl.rc >= res.optimal_rc - 1e-10)
                                break;

                            if (fl.vmask & bl.vmask) continue;
                            if (fl.load + bl.load > capacity) continue;

                            double total_rc = fl.rc + edge_cost + bl.rc;
                            if (total_rc < res.optimal_rc) {
                                res.optimal_rc = total_rc;
                                best_fi = fi;
                                best_bi = bi;
                            }
                            if (total_rc < -1e-6) res.neg_rc_count++;
                        }
                    }

                    if (++merge_check >= 256) {
                        merge_check = 0;
                        auto now = std::chrono::high_resolution_clock::now();
                        if (std::chrono::duration<double, std::milli>(now - t0).count() > time_limit_ms)
                            res.timed_out = true;
                    }
                }
            }

            if (best_fi >= 0) {
                // Reconstruct: fwd part [depot,...,i] + bwd part [j,...,depot]
                std::vector<int> path;
                {
                    std::vector<int> seg;
                    for (int t = best_fi; t >= 0; t = fwd_ph.labels[t].prev)
                        seg.push_back(fwd_ph.labels[t].node);
                    std::reverse(seg.begin(), seg.end());
                    for (int v : seg) path.push_back(v);
                }
                {
                    for (int t = best_bi; t >= 0; t = bwd_ph.labels[t].prev)
                        path.push_back(bwd_ph.labels[t].node);
                }

                res.optimal_path = std::move(path);
                int cost = 0;
                for (int k = 0; k + 1 < (int)res.optimal_path.size(); k++)
                    cost += dist[res.optimal_path[k]][res.optimal_path[k + 1]];
                res.optimal_cost = cost;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

} // namespace espprc_exact_modular
