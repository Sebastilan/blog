#pragma once
// ============================================================================
// ESPPRC k-cycle Elimination — Label-Setting with Sliding-Window Memory
// ============================================================================
// Prohibits revisiting any node that was visited within the last K-1 steps.
// This eliminates all cycles of length <= K.
//
// K=2: no immediate backtrack  (recent window size 1)
// K=3: no revisit within 2 steps (recent window size 2)
// K=5: no revisit within 4 steps (recent window size 4)
//
// State: (node, recent[K-1])
//   recent[0] = immediate predecessor
//   recent[K-2] = node visited K-1 steps ago
//   recent[i] = -1 if path is shorter than i+1
//
// Dominance: L1 dominates L2 iff:
//   L1.node == L2.node  AND  L1.recent == L2.recent
//   AND  L1.rc <= L2.rc  AND  L1.load <= L2.load
//
// State space: O(n^K) per direction (polynomial vs ng-route's O(n * 2^delta)).
// For small K, this is manageable; for large K (e.g., K=n), equivalent to exact.
//
// Note: k-cycle elimination gives a LOWER BOUND on exact ESPPRC RC,
// but tighter than ng-route because the constraint is on path recency not
// spatial proximity. However for typical CVRP instances ng-route Δ=8 tends
// to give tighter LP bounds with less computation than k-cycle K=5.
// ============================================================================

#include <vector>
#include <deque>
#include <array>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <unordered_map>
#include <numeric>

namespace espprc_kcycle {

struct Result {
    double optimal_rc;
    int    optimal_cost;
    std::vector<int> optimal_path;
    bool   path_is_elementary;
    int    labels_created;
    double time_ms;
    bool   timed_out;
    int    k_value;
};

// ── State key: hash of (node, recent[]) ─────────────────────────────────────
// Uses polynomial hashing; collision risk negligible for n<=63, K<=5.
template<int RSIZE>
inline uint64_t state_key(int node, const std::array<int, RSIZE>& recent) {
    uint64_t h = (uint64_t)(node + 1);
    for (int i = 0; i < RSIZE; i++)
        h = h * 100003ULL + (uint64_t)(recent[i] + 2);  // +2 so -1 maps to 1
    return h;
}

// ── Main solver ───────────────────────────────────────────────────────────────
template<int K>
Result solve(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    double time_limit_ms = 60000.0)
{
    static_assert(K >= 2, "K must be >= 2 for k-cycle elimination");

    auto t0 = std::chrono::high_resolution_clock::now();

    int n_cust = (int)cover_duals.size();
    int N = n_cust + 1;

    // Reduced cost matrix
    std::vector<double> c_bar(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c_bar[i * N + j] = (double)dist[i][j];
            if (j >= 1) c_bar[i * N + j] -= cover_duals[j - 1];
        }

    // Recent window: K-1 slots. recent[0] = immediate predecessor.
    constexpr int RSIZE = K - 1;

    struct Label {
        double rc;
        int    load;
        int    node;
        int    prev;   // index into labels[]
        std::array<int, RSIZE> recent;
    };

    std::vector<Label> labels;
    labels.reserve(16384);

    // Group labels by state key for O(bucket_size) dominance check
    // (labels in the same bucket have identical (node, recent[]))
    std::unordered_map<uint64_t, std::vector<int>> state_map;
    state_map.reserve(4096);

    // Root label at depot: recent all -1
    Label root;
    root.rc = 0.0; root.load = 0; root.node = 0; root.prev = -1;
    root.recent.fill(-1);
    labels.push_back(root);

    uint64_t root_key = state_key<RSIZE>(0, root.recent);
    state_map[root_key].push_back(0);

    std::deque<int> queue;
    queue.push_back(0);

    double best_rc    = 1e30;
    int    best_label = -1;
    bool   timed_out  = false;
    int    check_count = 0;

    // Dominance check: L1 dom L2 iff same bucket AND rc1 <= rc2 AND load1 <= load2
    auto is_dominated = [&](int node,
                            double rc, int load,
                            const std::array<int, RSIZE>& recent) -> bool {
        uint64_t key = state_key<RSIZE>(node, recent);
        auto it = state_map.find(key);
        if (it == state_map.end()) return false;
        for (int li : it->second) {
            const Label& el = labels[li];
            if (el.rc <= rc + 1e-10 && el.load <= load)
                return true;
        }
        return false;
    };

    // Add label: check dominance, remove dominated, insert
    auto add_label = [&](int node, double rc, int load,
                         const std::array<int, RSIZE>& recent, int prev) -> int {
        if (is_dominated(node, rc, load, recent)) return -1;

        uint64_t key = state_key<RSIZE>(node, recent);
        auto& bucket = state_map[key];

        // Remove labels dominated by the new label
        int w = 0;
        for (int k2 = 0; k2 < (int)bucket.size(); k2++) {
            int li = bucket[k2];
            const Label& el = labels[li];
            // el is dominated by new iff rc_new <= el.rc AND load_new <= el.load
            if (!(rc <= el.rc + 1e-10 && load <= el.load))
                bucket[w++] = li;
        }
        bucket.resize(w);

        int idx = (int)labels.size();
        labels.push_back({rc, load, node, prev, recent});
        bucket.push_back(idx);
        return idx;
    };

    while (!queue.empty()) {
        int li = queue.front();
        queue.pop_front();

        // Timeout check every 4096 iterations
        if (++check_count >= 4096) {
            check_count = 0;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(now - t0).count();
            if (elapsed > time_limit_ms) { timed_out = true; break; }
        }

        // Staleness check: label may have been dominated and removed from bucket
        {
            uint64_t key = state_key<RSIZE>(labels[li].node, labels[li].recent);
            auto it = state_map.find(key);
            if (it == state_map.end()) continue;
            bool found = false;
            for (int a : it->second) if (a == li) { found = true; break; }
            if (!found) continue;
        }

        // Copy all fields before extension loop: labels.push_back() inside
        // add_label() may reallocate the vector, invalidating any references.
        int    cur_node   = labels[li].node;
        double cur_rc     = labels[li].rc;
        int    cur_load   = labels[li].load;
        std::array<int, RSIZE> cur_recent = labels[li].recent;  // copy, not ref

        for (int j = 1; j <= n_cust; j++) {
            // k-cycle check: j must not appear in the recent window
            bool blocked = false;
            for (int r = 0; r < RSIZE; r++) {
                if (cur_recent[r] == j) { blocked = true; break; }
            }
            if (blocked) continue;

            int new_load = cur_load + demand[j];
            if (new_load > capacity) continue;

            double new_rc = cur_rc + c_bar[cur_node * N + j];

            // Sliding window: push cur_node, drop oldest
            std::array<int, RSIZE> new_recent;
            new_recent[0] = cur_node;
            for (int r = 1; r < RSIZE; r++)
                new_recent[r] = cur_recent[r - 1];

            int new_li = add_label(j, new_rc, new_load, new_recent, li);
            if (new_li >= 0) queue.push_back(new_li);
        }

        // Attempt return to depot
        if (cur_node != 0) {
            double final_rc = cur_rc + c_bar[cur_node * N + 0];
            if (final_rc < best_rc) {
                best_rc    = final_rc;
                best_label = li;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    Result res;
    res.labels_created = (int)labels.size();
    res.time_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.timed_out = timed_out;
    res.k_value  = K;

    if (best_label >= 0) {
        res.optimal_rc = best_rc;

        // Reconstruct path
        std::vector<int> path;
        for (int t = best_label; t >= 0; t = labels[t].prev)
            path.push_back(labels[t].node);
        std::reverse(path.begin(), path.end());
        path.push_back(0);
        res.optimal_path = path;

        // Elementary check
        std::vector<bool> seen(n_cust + 1, false);
        res.path_is_elementary = true;
        for (int v : res.optimal_path) {
            if (v == 0) continue;
            if (seen[v]) { res.path_is_elementary = false; break; }
            seen[v] = true;
        }

        int cost = 0;
        for (int k2 = 0; k2 + 1 < (int)res.optimal_path.size(); k2++)
            cost += dist[res.optimal_path[k2]][res.optimal_path[k2 + 1]];
        res.optimal_cost = cost;
    } else {
        res.optimal_rc = 1e30;
        res.optimal_cost = 0;
        res.path_is_elementary = true;
    }

    return res;
}

} // namespace espprc_kcycle
