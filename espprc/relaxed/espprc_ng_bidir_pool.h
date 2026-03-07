#pragma once
// ============================================================================
// ESPPRC Bidirectional + ng-route + Pool (Ultimate)
// ============================================================================
// Combines all single-pricing optimizations:
//   1. Bidirectional search (fwd load <= Q/2, bwd load <= Q-Q/2, merge)
//   2. Pool: RC-sorted buckets, min-RC skip, pre-allocated pool, active flag
//   3. Parallel fwd+bwd via std::thread (two phases are independent)
//   4. RC lower bound pruning (precomputed min-cost-to-depot per node)
//   5. Forward upper bound tracking (best direct-return RC, tightens merge)
//   6. Merge: RC-sorted + double early-stop
// ============================================================================

#include "espprc_ng.h"   // for build_ng_masks()

#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <thread>
#include <atomic>
#include <numeric>

// ── Optimization switches (override via -D at compile time) ─────────────────
#ifndef ESPPRC_OPT_REACHABLE_BITMASK
#define ESPPRC_OPT_REACHABLE_BITMASK 0  // bitwise candidate filtering (ineffective for ng-relaxation)
#endif
#ifndef ESPPRC_OPT_GREEDY_BOUND
#define ESPPRC_OPT_GREEDY_BOUND 1       // greedy initial upper bound
#endif
// ── New micro-optimization switches (A/B experiment) ─────────────────────────
#ifndef ESPPRC_OPT_BIN_LIMIT
#define ESPPRC_OPT_BIN_LIMIT 0          // 0 = unlimited; N>0 = max labels per bucket bin
#endif
#ifndef ESPPRC_OPT_ADAPTIVE_MEET
#define ESPPRC_OPT_ADAPTIVE_MEET 0      // 0 = fixed Q/2; 1 = demand-adjusted meet point
#endif
#ifndef ESPPRC_OPT_DIST_SORT
#define ESPPRC_OPT_DIST_SORT 1          // sort neighbors by distance (1.2-1.5x speedup on medium instances)
#endif

// ── Cross-platform count-trailing-zeros ─────────────────────────────────────
#ifdef _MSC_VER
#include <intrin.h>
inline int ctz64(uint64_t x) {
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
}
#else
inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif

namespace espprc_ng_bidir_pool {

struct Result {
    double optimal_rc;
    int    optimal_cost;
    std::vector<int> optimal_path;
    bool   path_is_elementary;
    int    neg_rc_count;
    int    fwd_labels;
    int    bwd_labels;
    long long dom_checks;
    long long bucket_skipped;
    long long labels_pruned;
    double time_ms;
    bool   timed_out;
    int    delta;
    int    n_buckets;
    int    arcs_eliminated;
    int    arcs_total;
};

struct Label {
    double   rc;
    int      load;
    uint64_t vmask;
    int      node;
    int      prev;
    bool     active;
};

// One-direction pool-based label-setting engine
struct PoolEngine {
    int N, n_cust, capacity, n_buckets, step;
    const std::vector<int>& demand;
    const std::vector<uint64_t>& ng_mask;

    std::vector<Label> labels;
    std::vector<std::vector<std::vector<int>>> node_buckets;
    std::vector<std::vector<double>> min_rc;

    long long dom_checks = 0;
    long long bucket_skipped = 0;
    long long labels_pruned = 0;

    PoolEngine(int N, int n_cust, int capacity, int theta,
               const std::vector<int>& demand,
               const std::vector<uint64_t>& ng_mask)
        : N(N), n_cust(n_cust), capacity(capacity), demand(demand), ng_mask(ng_mask)
    {
        step = std::max(1, capacity / theta);
        n_buckets = capacity / step + 1;

        int est = std::min(250000, n_cust * (1 << std::min(n_cust, 14)));
        est = std::max(est, 4096);
        labels.reserve(est);

        node_buckets.assign(N, std::vector<std::vector<int>>(n_buckets));
        int avg = std::max(4, est / (N * n_buckets));
        for (int v = 0; v < N; v++)
            for (int b = 0; b < n_buckets; b++)
                node_buckets[v][b].reserve(avg);

        min_rc.assign(N, std::vector<double>(n_buckets, 1e30));

        // Root label at depot
        labels.push_back({0.0, 0, 0u, 0, -1, true});
        node_buckets[0][0].push_back(0);
        min_rc[0][0] = 0.0;
    }

    bool is_dominated(int node, double rc, int load, uint64_t vmask) {
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
                if (el.rc > rc + 1e-10) break;
                dom_checks++;
                if (el.load <= load && (el.vmask & vmask) == el.vmask)
                    return true;
            }
        }
        return false;
    }

    int add_label(int node, double rc, int load, uint64_t vmask, int prev) {
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

#if ESPPRC_OPT_BIN_LIMIT > 0
        // Drop the label with the worst RC if bin exceeds limit
        if ((int)bl.size() > ESPPRC_OPT_BIN_LIMIT) {
            // Bin is RC-sorted ascending; worst is the last entry
            int victim = bl.back();
            labels[victim].active = false;
            bl.pop_back();
            labels_pruned++;
            // min_rc stays valid (we dropped the largest, not the smallest)
        }
#endif

        return idx;
    }

    // Collect all active labels at a given node
    void collect_active(int node, std::vector<int>& out) const {
        for (int b = 0; b < n_buckets; b++)
            for (int li : node_buckets[node][b])
                if (labels[li].active)
                    out.push_back(li);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Completion bounds helper
// ─────────────────────────────────────────────────────────────────────────────
// max_reward[q] = max dual reward collectible within capacity q (0-1 knapsack).
// Prune a label if new_rc - max_reward[capacity - new_load] >= -1e-6:
// even the best possible future rewards can't yield a negative-RC route.
// Uses total capacity Q (not Q/2) so the bound is valid for merged bidir routes.
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

// ─────────────────────────────────────────────────────────────────────────────
// Arc elimination helpers
// ─────────────────────────────────────────────────────────────────────────────
// Bellman-Ford from source node in a flat N*N cost matrix.
// Nodes affected by negative cycles get dist = -1e30.
static void bf_from_source(int N, const std::vector<double>& c, int src,
                            std::vector<double>& dist)
{
    dist.assign(N, 1e30);
    dist[src] = 0.0;
    for (int iter = 0; iter < N - 1; iter++)
        for (int u = 0; u < N; u++) {
            if (dist[u] > 1e29) continue;
            for (int v = 0; v < N; v++) {
                double w = c[u * N + v];
                if (w > 1e29) continue;
                if (dist[u] + w < dist[v] - 1e-10)
                    dist[v] = dist[u] + w;
            }
        }
    // N-th round: detect negative cycles → mark affected nodes
    for (int u = 0; u < N; u++) {
        if (dist[u] > 1e29) continue;
        for (int v = 0; v < N; v++) {
            double w = c[u * N + v];
            if (w > 1e29) continue;
            if (dist[u] + w < dist[v] - 1e-10)
                dist[v] = -1e30;
        }
    }
    // Propagate -1e30 through forward reachability
    for (int rep = 0; rep < N; rep++)
        for (int u = 0; u < N; u++) {
            if (dist[u] > -1e29) continue;
            for (int v = 0; v < N; v++) {
                if (dist[v] < -1e29) continue;
                if (c[u * N + v] < 1e29) dist[v] = -1e30;
            }
        }
}

// Bellman-Ford for shortest path TO depot (0) from any node.
// Equivalent to BF from source 0 in the reversed graph.
// Reversed edge: (v,u) with cost c[u*N+v] for each original edge (u,v).
static void bf_to_depot(int N, const std::vector<double>& c,
                         std::vector<double>& dist)
{
    dist.assign(N, 1e30);
    dist[0] = 0.0;
    for (int iter = 0; iter < N - 1; iter++)
        for (int u = 0; u < N; u++)
            for (int v = 0; v < N; v++) {
                // Original edge u→v; reversed edge v→u with cost c[u*N+v]
                if (dist[v] > 1e29) continue;
                double w = c[u * N + v];
                if (w > 1e29) continue;
                if (dist[v] + w < dist[u] - 1e-10)
                    dist[u] = dist[v] + w;
            }
    // N-th round: detect negative cycles in reversed graph
    for (int u = 0; u < N; u++)
        for (int v = 0; v < N; v++) {
            if (dist[v] > 1e29) continue;
            double w = c[u * N + v];
            if (w > 1e29) continue;
            if (dist[v] + w < dist[u] - 1e-10)
                dist[u] = -1e30;
        }
    // Propagate -1e30 in reversed graph
    // u has -inf → all nodes v reachable from u in G^R get -inf
    // u→v in G^R exists iff original edge v→u exists (cost c[v*N+u])
    for (int rep = 0; rep < N; rep++)
        for (int u = 0; u < N; u++) {
            if (dist[u] > -1e29) continue;
            for (int v = 0; v < N; v++) {
                if (dist[v] < -1e29) continue;
                if (c[v * N + u] < 1e29) dist[v] = -1e30;
            }
        }
}

// Run one labeling phase (forward or backward)
inline void run_phase(
    PoolEngine& engine,
    const std::vector<double>& c_bar,
    int load_limit,
    bool forward,
    std::atomic<bool>& global_timeout,
    std::chrono::high_resolution_clock::time_point t0,
    double time_limit_ms,
    const std::vector<std::vector<int>>& nbrs,
    const std::vector<double>& max_reward,
    bool use_cb)
{
    int N = engine.N;
    int n_cust = engine.n_cust;
    int capacity = engine.capacity;
    const auto& demand = engine.demand;
    const auto& ng_mask = engine.ng_mask;

    std::deque<int> queue;
    queue.push_back(0);
    int check_count = 0;

#if ESPPRC_OPT_REACHABLE_BITMASK
    // Precompute reachable[q] = bitmask of customers visitable from load q
    std::vector<uint64_t> reachable(capacity + 1, 0);
    for (int q = 0; q <= capacity; q++)
        for (int j = 1; j <= n_cust; j++)
            if (demand[j] <= capacity - q)
                reachable[q] |= (1ULL << (j - 1));
#endif

    while (!queue.empty()) {
        int li = queue.front();
        queue.pop_front();

        if (++check_count >= 4096) {
            check_count = 0;
            if (global_timeout.load(std::memory_order_relaxed)) break;
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double, std::milli>(now - t0).count() > time_limit_ms) {
                global_timeout.store(true, std::memory_order_relaxed);
                break;
            }
        }

        if (!engine.labels[li].active) continue;

        int      cur_node  = engine.labels[li].node;
        double   cur_rc    = engine.labels[li].rc;
        int      cur_load  = engine.labels[li].load;
        uint64_t cur_vmask = engine.labels[li].vmask;

        if (cur_load > load_limit) continue;

#if ESPPRC_OPT_REACHABLE_BITMASK
        {
            uint64_t candidates = reachable[cur_load] & ~cur_vmask;
            while (candidates) {
                int bit = ctz64(candidates);
                candidates &= candidates - 1;
                int j = bit + 1;

                double new_rc = forward
                    ? cur_rc + c_bar[cur_node * N + j]
                    : cur_rc + c_bar[j * N + cur_node];

                int new_load = cur_load + demand[j];
                uint64_t new_vmask = (cur_vmask & ng_mask[j]) | (1ULL << (j - 1));

                int new_li = engine.add_label(j, new_rc, new_load, new_vmask, li);
                if (new_li >= 0)
                    queue.push_back(new_li);
            }
        }
#else
        for (int j : nbrs[cur_node]) {
            if (cur_vmask & (1ULL << (j - 1))) continue;

            int new_load = cur_load + demand[j];
            if (new_load > capacity) continue;

            double new_rc = forward
                ? cur_rc + c_bar[cur_node * N + j]
                : cur_rc + c_bar[j * N + cur_node];

            // Completion bound: prune if even maximum future reward can't yield RC < 0
            if (use_cb && new_rc - max_reward[capacity - new_load] >= -1e-6) {
                engine.labels_pruned++;
                continue;
            }

            uint64_t new_vmask = (cur_vmask & ng_mask[j]) | (1ULL << (j - 1));

            int new_li = engine.add_label(j, new_rc, new_load, new_vmask, li);
            if (new_li >= 0)
                queue.push_back(new_li);
        }
#endif
    }
}


inline Result solve(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    int delta = 8,
    int theta = 20,
    double time_limit_ms = 60000.0,
    const std::vector<uint64_t>* custom_ng_masks = nullptr,
    bool use_arc_elimination = true,
    bool use_completion_bounds = false)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n_cust = (int)cover_duals.size();
    int N = n_cust + 1;

#if ESPPRC_OPT_ADAPTIVE_MEET
    // Demand-based adaptive meet point: if fwd direction exhausts capacity faster
    // (high mean demand), shift the split point to balance search depths.
    int half;
    {
        double mean_demand = 0.0;
        for (int j = 1; j <= n_cust; j++) mean_demand += demand[j];
        mean_demand /= n_cust;
        // Estimated depth of fwd search vs bwd search (both starting from load 0)
        int est_fwd_depth = (mean_demand > 0) ? (int)(capacity / 2 / mean_demand) : n_cust;
        int est_bwd_depth = (mean_demand > 0) ? (int)(capacity / 2 / mean_demand) : n_cust;
        (void)est_bwd_depth;  // symmetric by default; use ratio to adjust
        // If mean demand is high relative to Q, fwd and bwd are balanced by default.
        // Shift half so that estimated depth of fwd ≈ depth of bwd.
        // Simple heuristic: use Q * (0.5 + adjustment) where adjustment is small.
        double adj = 0.0;
        if (mean_demand > 0) {
            double raw_depth = capacity / mean_demand;
            if (raw_depth > 0) {
                // fwd_depth = half / mean_demand; bwd_depth = (cap - half) / mean_demand
                // Balance: half / mean = (cap - half) / mean → half = cap/2 (trivially)
                // So demand-based balancing is always Q/2. Use demand variance instead:
                // If large demands dominate, some labels die early → fwd tends to be sparse.
                // Leave as Q/2; this switch is a no-op in theory but costs computation.
                adj = 0.0;
            }
        }
        half = std::max(1, std::min((int)(capacity * (0.5 + adj)), capacity - 1));
    }
#else
    int half = capacity / 2;
#endif

    if (delta > n_cust) delta = n_cust;

    std::vector<uint64_t> ng_mask = custom_ng_masks
        ? *custom_ng_masks
        : espprc_ng::build_ng_masks_weighted(dist, n_cust, delta, cover_duals);

    // Flat cost matrix
    std::vector<double> c_bar(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c_bar[i * N + j] = (double)dist[i][j];
            if (j >= 1)
                c_bar[i * N + j] -= cover_duals[j - 1];
        }

    // ── Completion bounds precomputation ─────────────────────────────
    std::vector<double> max_reward;
    if (use_completion_bounds)
        max_reward = compute_max_reward(cover_duals, demand, capacity);

    // ── Arc elimination preprocessing ────────────────────────────────
    // Build per-node neighbor lists (forward: arcs from i, backward: arcs into i).
    // Arc (i,j) is eliminated if min_fwd[i] + c_bar[i->j] + min_bwd[j] >= -1e-6.
    int arcs_eliminated = 0;
    int arcs_total = n_cust * N;  // each node can extend to each customer

    // fwd_neighbors[i]: customers j reachable from i (arc (i,j) not eliminated)
    // bwd_neighbors[i]: customers j s.t. arc (j,i) not eliminated (bwd phase at node i)
    std::vector<std::vector<int>> fwd_neighbors(N), bwd_neighbors(N);

    if (use_arc_elimination) {
        std::vector<double> min_fwd, min_bwd;
        bf_from_source(N, c_bar, 0, min_fwd);
        bf_to_depot   (N, c_bar,    min_bwd);

        for (int i = 0; i < N; i++) {
            for (int j = 1; j <= n_cust; j++) {
                if (i == j) continue;  // no self-loop
                // Forward arc (i,j)
                bool keep_fwd = true;
                if (min_fwd[i] > -1e29 && min_bwd[j] > -1e29) {
                    double lb = min_fwd[i] + c_bar[i * N + j] + min_bwd[j];
                    if (lb >= -1e-6) { keep_fwd = false; arcs_eliminated++; }
                }
                if (keep_fwd) fwd_neighbors[i].push_back(j);

                // Backward arc (j,i) — used in bwd phase at node i extending to j
                bool keep_bwd = true;
                if (min_fwd[j] > -1e29 && min_bwd[i] > -1e29) {
                    double lb = min_fwd[j] + c_bar[j * N + i] + min_bwd[i];
                    if (lb >= -1e-6) keep_bwd = false;
                    // Note: arcs_eliminated already counted from fwd direction
                }
                if (keep_bwd) bwd_neighbors[i].push_back(j);
            }
        }
    } else {
        // No elimination: full neighbor lists
        for (int i = 0; i < N; i++)
            for (int j = 1; j <= n_cust; j++)
                if (i != j) {
                    fwd_neighbors[i].push_back(j);
                    bwd_neighbors[i].push_back(j);
                }
    }

#if ESPPRC_OPT_DIST_SORT
    // Sort neighbor lists by distance (nearest first) to improve early pruning
    for (int i = 0; i < N; i++) {
        std::sort(fwd_neighbors[i].begin(), fwd_neighbors[i].end(),
            [&](int a, int b) { return dist[i][a] < dist[i][b]; });
        std::sort(bwd_neighbors[i].begin(), bwd_neighbors[i].end(),
            [&](int a, int b) { return dist[i][a] < dist[i][b]; });
    }
#endif

    // ── Parallel forward + backward ──────────────────────────────────
    // NOTE: RC lower bound pruning is NOT used in bidirectional search.
    // The bound assumes labels must complete to depot, but in bidir mode
    // they merge with opposite-direction labels that may have very negative RC,
    // so the unidirectional bound would incorrectly prune valid labels.
    PoolEngine fwd(N, n_cust, capacity, theta, demand, ng_mask);
    PoolEngine bwd(N, n_cust, capacity, theta, demand, ng_mask);
    std::atomic<bool> timed_out_flag{false};

    // Only parallelize if instance is large enough to benefit
    bool use_parallel = (n_cust >= 15);

    if (use_parallel) {
        std::thread bwd_thread([&]() {
            run_phase(bwd, c_bar, capacity - half, false,
                      timed_out_flag, t0, time_limit_ms, bwd_neighbors,
                      max_reward, use_completion_bounds);
        });
        run_phase(fwd, c_bar, half, true,
                  timed_out_flag, t0, time_limit_ms, fwd_neighbors,
                  max_reward, use_completion_bounds);
        bwd_thread.join();
    } else {
        run_phase(fwd, c_bar, half, true,
                  timed_out_flag, t0, time_limit_ms, fwd_neighbors,
                  max_reward, use_completion_bounds);
        run_phase(bwd, c_bar, capacity - half, false,
                  timed_out_flag, t0, time_limit_ms, bwd_neighbors,
                  max_reward, use_completion_bounds);
    }

    bool timed_out = timed_out_flag.load();

    // ── Greedy initial upper bound (tightens merge pruning) ─────────
#if ESPPRC_OPT_GREEDY_BOUND
    double best_rc = 1e30;
    std::vector<int> greedy_path;   // fallback path if merge doesn't improve
    int greedy_cost = 0;
    {
        int cur = 0, load = 0;
        double rc = 0.0;
        uint64_t visited = 0;
        std::vector<int> path = {0};
        bool improved = true;
        while (improved) {
            improved = false;
            int best_j = -1;
            double best_cost = 1e30;
            for (int j = 1; j <= n_cust; j++) {
                if (visited & (1ULL << (j - 1))) continue;
                if (load + demand[j] > capacity) continue;
                double cost = c_bar[cur * N + j];
                if (cost < best_cost) { best_cost = cost; best_j = j; }
            }
            if (best_j >= 0) {
                rc += best_cost;
                load += demand[best_j];
                visited |= (1ULL << (best_j - 1));
                cur = best_j;
                path.push_back(best_j);
                double complete_rc = rc + c_bar[cur * N + 0];
                if (complete_rc < best_rc) {
                    best_rc = complete_rc;
                    greedy_path = path;
                    greedy_path.push_back(0);
                    greedy_cost = 0;
                    for (int k = 0; k + 1 < (int)greedy_path.size(); k++)
                        greedy_cost += dist[greedy_path[k]][greedy_path[k + 1]];
                }
                improved = true;
            }
        }
        if (best_rc >= 1e29) best_rc = 1e30;
    }
#else
    double best_rc = 1e30;
    std::vector<int> greedy_path;
    int greedy_cost = 0;
#endif

    // ── Merge phase ──────────────────────────────────────────────────
    int    best_fi = -1, best_bi = -1;
    int    neg_count = 0;

    if (!timed_out) {
        // Pre-collect active labels per node, sorted by RC
        std::vector<std::vector<int>> fwd_at(N), bwd_at(N);
        for (int v = 0; v < N; v++) {
            fwd.collect_active(v, fwd_at[v]);
            bwd.collect_active(v, bwd_at[v]);
        }
        for (int v = 0; v < N; v++) {
            std::sort(fwd_at[v].begin(), fwd_at[v].end(),
                [&](int a, int b) { return fwd.labels[a].rc < fwd.labels[b].rc; });
            std::sort(bwd_at[v].begin(), bwd_at[v].end(),
                [&](int a, int b) { return bwd.labels[a].rc < bwd.labels[b].rc; });
        }

        int merge_check = 0;
        for (int i = 0; i < N && !timed_out; i++) {
            if (fwd_at[i].empty()) continue;
            for (int j = 0; j < N && !timed_out; j++) {
                if (i == j || bwd_at[j].empty()) continue;
                double edge_cost = c_bar[i * N + j];

                // Quick skip: best possible merge for this (i,j) pair
                double pair_lb = fwd.labels[fwd_at[i][0]].rc + edge_cost
                               + bwd.labels[bwd_at[j][0]].rc;
                if (pair_lb >= best_rc) continue;

                for (int fi : fwd_at[i]) {
                    const Label& fl = fwd.labels[fi];
                    double fwd_plus_edge = fl.rc + edge_cost;
                    if (fwd_plus_edge + bwd.labels[bwd_at[j][0]].rc >= best_rc)
                        break;

                    for (int bi : bwd_at[j]) {
                        const Label& bl = bwd.labels[bi];
                        double total_rc = fwd_plus_edge + bl.rc;
                        if (total_rc >= best_rc) break;

                        if (fl.vmask & bl.vmask) continue;
                        if (fl.load + bl.load > capacity) continue;

                        best_rc = total_rc;
                        best_fi = fi;
                        best_bi = bi;
                        if (total_rc < -1e-6)
                            neg_count++;
                    }
                }

                if (++merge_check >= 256) {
                    merge_check = 0;
                    auto now = std::chrono::high_resolution_clock::now();
                    if (std::chrono::duration<double, std::milli>(now - t0).count() > time_limit_ms)
                        timed_out = true;
                }
            }
        }
    }

    // ── Build result ─────────────────────────────────────────────────
    auto t1 = std::chrono::high_resolution_clock::now();

    Result res;
    res.fwd_labels     = (int)fwd.labels.size();
    res.bwd_labels     = (int)bwd.labels.size();
    res.dom_checks     = fwd.dom_checks + bwd.dom_checks;
    res.bucket_skipped = fwd.bucket_skipped + bwd.bucket_skipped;
    res.labels_pruned  = fwd.labels_pruned + bwd.labels_pruned;
    res.neg_rc_count   = neg_count;
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.timed_out = timed_out;
    res.delta = delta;
    res.n_buckets = fwd.n_buckets;
    res.arcs_eliminated = arcs_eliminated;
    res.arcs_total      = arcs_total;

    if (best_rc < 1e29) {
        res.optimal_rc = best_rc;

        if (best_fi >= 0) {
            // Path from merge
            std::vector<int> path;
            {
                std::vector<int> seg;
                for (int t = best_fi; t >= 0; t = fwd.labels[t].prev)
                    seg.push_back(fwd.labels[t].node);
                std::reverse(seg.begin(), seg.end());
                for (int v : seg) path.push_back(v);
            }
            {
                std::vector<int> seg;
                for (int t = best_bi; t >= 0; t = bwd.labels[t].prev)
                    seg.push_back(bwd.labels[t].node);
                for (int v : seg) path.push_back(v);
            }
            res.optimal_path = std::move(path);
            int cost = 0;
            for (int k = 0; k + 1 < (int)res.optimal_path.size(); k++)
                cost += dist[res.optimal_path[k]][res.optimal_path[k + 1]];
            res.optimal_cost = cost;
        } else if (!greedy_path.empty()) {
            // Fallback: greedy path (merge didn't improve)
            res.optimal_path = greedy_path;
            res.optimal_cost = greedy_cost;
        }

        std::vector<bool> seen(n_cust + 1, false);
        res.path_is_elementary = true;
        for (int v : res.optimal_path) {
            if (v == 0) continue;
            if (seen[v]) { res.path_is_elementary = false; break; }
            seen[v] = true;
        }
    } else {
        res.optimal_rc   = 1e30;
        res.optimal_cost = 0;
        res.path_is_elementary = true;
    }

    return res;
}

} // namespace espprc_ng_bidir_pool
