#pragma once
// Global column pool — stores all unique columns for cross-node reuse.
// Pool scan computes reduced cost from duals, returns negative-RC columns.

#include "types.h"
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <tuple>

class ColumnPool {
public:
    struct ScanHit {
        int cost;
        std::vector<int> visits;
        std::vector<int> path;
        size_t hash;
    };

    // Add column to pool (dedup by path hash). Returns true if new.
    bool add(int cost, const std::vector<int>& visits,
             const std::vector<int>& path, int n) {
        size_t h = path_hash(path);
        if (seen_.count(h)) return false;
        seen_.insert(h);

        Entry e;
        e.cost = cost;
        e.visits = visits;
        e.path = path;
        e.hash = h;
        for (int k = 0; k + 1 < (int)path.size(); k++)
            e.edges.insert(edge_key(path[k], path[k+1], n));
        entries_.push_back(std::move(e));
        return true;
    }

    // Scan pool: return columns with rc < -eps, not using any forbidden edge.
    std::vector<ScanHit> scan(
        const double* cover_duals, int n_customers,
        const std::map<int,double>& edge_duals,
        const std::set<int>& forbidden_keys) const
    {
        std::vector<ScanHit> hits;
        for (auto& e : entries_) {
            // Skip if uses any forbidden edge
            bool fbd = false;
            for (int ek : e.edges) {
                if (forbidden_keys.count(ek)) { fbd = true; break; }
            }
            if (fbd) continue;

            // Compute reduced cost: rc = cost - sum(pi_i) - sum(mu_e)
            double rc = (double)e.cost;
            for (int v : e.visits)
                rc -= cover_duals[v - 1];
            for (auto& [ek, d] : edge_duals) {
                if (e.edges.count(ek)) rc -= d;
            }
            if (rc < -1e-6)
                hits.push_back({e.cost, e.visits, e.path, e.hash});
        }
        return hits;
    }

    int size() const { return (int)entries_.size(); }

private:
    struct Entry {
        int cost;
        std::vector<int> visits;
        std::vector<int> path;
        size_t hash;
        std::set<int> edges;  // precomputed edge keys
    };

    std::vector<Entry> entries_;
    std::unordered_set<size_t> seen_;
};
