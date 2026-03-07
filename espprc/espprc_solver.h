#pragma once
// ============================================================================
// ESPPRC Unified Solver Interface
// ============================================================================
// Single entry point for all ESPPRC pricing variants:
//
//   Solver::BIDIR_POOL     ng-route relaxation (bidirectional + parallel, Pool)
//   Solver::EXACT_POOL     exact ESPPRC (full bitmask, Pool)
//   Solver::DSSR_POOL      exact ESPPRC (DSSR iterative tightening, Pool)
//
// Usage:
//   #include "espprc_solver.h"
//   auto r = espprc::solve(dist, demand, capacity, duals, espprc::Solver::BIDIR_POOL);
//   if (r.optimal_rc < -1e-6) { /* use r.optimal_path as new column */ }
//
// All solvers share the same input format:
//   dist[N][N]     integer distance matrix (N = n_customers + 1, node 0 = depot)
//   demand[N]      demand per node (demand[0] = 0)
//   capacity       vehicle capacity Q
//   cover_duals    dual values pi[1..n_cust] from covering constraints
//
// The reduced cost of arc (i,j) is: dist[i][j] - pi[j]  (pi[0] = 0)
// ============================================================================

#include "relaxed/espprc_ng_bidir_pool.h"
#include "exact/espprc_exact_pool.h"
#include "exact/espprc_dssr_pool.h"

#include <vector>
#include <string>

namespace espprc {

enum class Solver {
    BIDIR_POOL,     // L2 relaxation: ng-route, bidirectional + parallel
    EXACT_POOL,     // L3 exact: full bitmask (n <= 31)
    DSSR_POOL,      // L3 exact: DSSR iterative (n <= 31)
};

struct Result {
    // ── Core output (always valid) ──────────────────────────────────
    double optimal_rc;              // best reduced cost found (1e30 if none)
    int    optimal_cost;            // original cost of best path
    std::vector<int> optimal_path;  // node sequence [0, v1, ..., vk, 0]
    bool   path_is_elementary;      // true if no repeated customers
    bool   timed_out;               // true if time limit hit
    double time_ms;                 // wall-clock time

    // ── Diagnostics ─────────────────────────────────────────────────
    int       neg_rc_count;         // number of negative-RC paths found
    int       labels_total;         // total labels created
    long long dom_checks;           // dominance checks performed
    long long bucket_skipped;       // buckets skipped by min-RC filter
    int       delta;                // ng-neighborhood size used (0 for exact)
    int       dssr_iterations;      // DSSR iterations (0 for non-DSSR)
    int       critical_count;       // DSSR critical vertices (0 for non-DSSR)

    // ── Helpers ─────────────────────────────────────────────────────
    bool has_negative_rc() const { return optimal_rc < -1e-6; }
};

struct Config {
    int    delta          = 8;      // ng-neighborhood size (ignored by EXACT_POOL)
    int    theta          = 20;     // bucket granularity (capacity / theta)
    int    max_dssr_iter  = 20;     // max DSSR iterations (only for DSSR_POOL)
    double time_limit_ms  = 60000;  // time limit in milliseconds
    const std::vector<uint64_t>* custom_ng_masks = nullptr;  // per-node masks (BIDIR_POOL)
};

// ── Main solve function ─────────────────────────────────────────────
inline Result solve(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    Solver solver = Solver::BIDIR_POOL,
    const Config& cfg = Config{})
{
    Result res{};
    res.optimal_rc = 1e30;
    res.optimal_cost = 0;
    res.path_is_elementary = true;
    res.delta = cfg.delta;

    switch (solver) {

    case Solver::BIDIR_POOL: {
        auto r = espprc_ng_bidir_pool::solve(
            dist, demand, capacity, cover_duals,
            cfg.delta, cfg.theta, cfg.time_limit_ms,
            cfg.custom_ng_masks);
        res.optimal_rc        = r.optimal_rc;
        res.optimal_cost      = r.optimal_cost;
        res.optimal_path      = std::move(r.optimal_path);
        res.path_is_elementary = r.path_is_elementary;
        res.timed_out         = r.timed_out;
        res.time_ms           = r.time_ms;
        res.neg_rc_count      = r.neg_rc_count;
        res.labels_total      = r.fwd_labels + r.bwd_labels;
        res.dom_checks        = r.dom_checks;
        res.bucket_skipped    = r.bucket_skipped;
        res.delta             = r.delta;
        break;
    }

    case Solver::EXACT_POOL: {
        auto r = espprc_exact_pool::solve(
            dist, demand, capacity, cover_duals,
            cfg.theta, cfg.time_limit_ms);
        res.optimal_rc        = r.optimal_rc;
        res.optimal_cost      = r.optimal_cost;
        res.optimal_path      = std::move(r.optimal_path);
        res.path_is_elementary = r.path_is_elementary;
        res.timed_out         = r.timed_out;
        res.time_ms           = r.time_ms;
        res.neg_rc_count      = r.neg_rc_count;
        res.labels_total      = r.labels_created;
        res.dom_checks        = r.dom_checks;
        res.bucket_skipped    = r.bucket_skipped;
        res.delta             = 0;
        break;
    }

    case Solver::DSSR_POOL: {
        auto r = espprc_dssr_pool::solve(
            dist, demand, capacity, cover_duals,
            cfg.delta, cfg.theta, cfg.max_dssr_iter, cfg.time_limit_ms);
        res.optimal_rc        = r.optimal_rc;
        res.optimal_cost      = r.optimal_cost;
        res.optimal_path      = std::move(r.optimal_path);
        res.path_is_elementary = r.path_is_elementary;
        res.timed_out         = r.timed_out;
        res.time_ms           = r.time_ms;
        res.neg_rc_count      = r.neg_rc_count;
        res.labels_total      = r.labels_created;
        res.dom_checks        = r.dom_checks;
        res.bucket_skipped    = r.bucket_skipped;
        res.delta             = r.delta;
        res.dssr_iterations   = r.dssr_iterations;
        res.critical_count    = r.critical_count;
        break;
    }

    } // switch
    return res;
}

// ── Solver name string ──────────────────────────────────────────────
inline const char* solver_name(Solver s) {
    switch (s) {
        case Solver::BIDIR_POOL: return "bidir_pool";
        case Solver::EXACT_POOL: return "exact_pool";
        case Solver::DSSR_POOL:  return "dssr_pool";
        default:                 return "unknown";
    }
}

} // namespace espprc
