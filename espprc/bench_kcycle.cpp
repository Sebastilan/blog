// ============================================================================
// bench_kcycle.cpp — k-cycle Elimination vs ng-route vs Exact Comparison
// ============================================================================
// Runs 6 configurations on each testset instance and prints a TSV table:
//   kcycle_K2, kcycle_K3, kcycle_K5 (K5 only on n<=18)
//   ng_D5, ng_D8
//   exact (only on n<=18)
//
// Each configuration runs 3 times; timing = median.
// Match_Exact: YES if |RC - exact_RC| < 0.5, SKIP if not run.
//
// Usage:
//   bench_kcycle [instances_dir]
// ============================================================================

#include "relaxed/espprc_kcycle.h"
#include "relaxed/espprc_ng.h"
#include "exact/espprc.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Load JSON instance ────────────────────────────────────────────────────────
static bool load_instance(
    const std::string& path,
    std::string& name,
    std::vector<std::vector<int>>& dist,
    std::vector<int>& demand,
    int& capacity,
    std::vector<double>& cover_duals,
    double& oracle_rc)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    json j;
    try { f >> j; } catch (...) { return false; }

    name        = j.value("name", path);
    capacity    = j["capacity"].get<int>();
    demand      = j["demand"].get<std::vector<int>>();
    cover_duals = j["cover_duals"].get<std::vector<double>>();
    dist        = j["dist"].get<std::vector<std::vector<int>>>();

    oracle_rc = 1e30;
    if (j.contains("oracle") && !j["oracle"]["optimal_rc"].is_null())
        oracle_rc = j["oracle"]["optimal_rc"].get<double>();

    return true;
}

// ── Run k-cycle (templated) 3 times, return median ───────────────────────────
template<int K>
static std::pair<double, double> run_kcycle(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    double time_limit_ms = 60000.0)
{
    double times[3], rcs[3];
    for (int r = 0; r < 3; r++) {
        auto res = espprc_kcycle::solve<K>(dist, demand, capacity, cover_duals, time_limit_ms);
        times[r] = res.time_ms;
        rcs[r]   = res.optimal_rc;
    }
    // Median
    double t[3] = {times[0], times[1], times[2]};
    std::sort(t, t + 3);
    double median_time = t[1];
    // Use RC from run with median time
    int mid_idx = 0;
    for (int r = 0; r < 3; r++) if (std::abs(times[r] - median_time) < 1e-9) { mid_idx = r; break; }
    return {median_time, rcs[mid_idx]};
}

// ── Run ng-route 3 times, return median ──────────────────────────────────────
static std::pair<double, double> run_ng(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    int delta,
    double time_limit_ms = 60000.0)
{
    double times[3], rcs[3];
    for (int r = 0; r < 3; r++) {
        auto res = espprc_ng::solve(dist, demand, capacity, cover_duals, delta, time_limit_ms);
        times[r] = res.time_ms;
        rcs[r]   = res.optimal_rc;
    }
    double t[3] = {times[0], times[1], times[2]};
    std::sort(t, t + 3);
    double median_time = t[1];
    int mid_idx = 0;
    for (int r = 0; r < 3; r++) if (std::abs(times[r] - median_time) < 1e-9) { mid_idx = r; break; }
    return {median_time, rcs[mid_idx]};
}

// ── Run exact 3 times, return median ─────────────────────────────────────────
static std::pair<double, double> run_exact(
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    double time_limit_ms = 60000.0)
{
    double times[3], rcs[3];
    for (int r = 0; r < 3; r++) {
        auto res = espprc::solve(dist, demand, capacity, cover_duals, time_limit_ms);
        times[r] = res.time_ms;
        rcs[r]   = res.optimal_rc;
    }
    double t[3] = {times[0], times[1], times[2]};
    std::sort(t, t + 3);
    double median_time = t[1];
    int mid_idx = 0;
    for (int r = 0; r < 3; r++) if (std::abs(times[r] - median_time) < 1e-9) { mid_idx = r; break; }
    return {median_time, rcs[mid_idx]};
}

// ── Match check: YES if diff < 0.5, else WORSE/SKIP ─────────────────────────
static std::string match(double rc, double exact_rc) {
    if (exact_rc > 1e29) return "SKIP";
    double diff = rc - exact_rc;   // positive = relaxation gave worse RC (higher) = shouldn't happen for valid relaxation
    // k-cycle and ng-route are relaxations: their RC <= exact RC (tighter routes)
    if (diff > 0.5)  return "WORSE";   // unexpected
    if (std::abs(diff) < 0.5) return "YES";
    return "NO";
}

// ── Print helpers ─────────────────────────────────────────────────────────────
static void print_row(const std::string& inst, int n, const std::string& config,
                      double time_ms, double rc, const std::string& match_str)
{
    std::cout << inst
              << "\t" << n
              << "\t" << config
              << "\t";
    if (time_ms >= 59000)
        std::cout << "TIMEOUT";
    else
        std::cout << std::fixed << std::setprecision(3) << time_ms;
    std::cout << "\t";
    if (rc > 1e29)
        std::cout << "NONE";
    else
        std::cout << std::fixed << std::setprecision(4) << rc;
    std::cout << "\t" << match_str << "\n";
}

int main(int argc, char* argv[]) {
    std::string instances_dir = (argc > 1) ? argv[1] : "../testset/instances";

    if (!fs::exists(instances_dir)) {
        std::cerr << "Instances dir not found: " << instances_dir << "\n";
        return 1;
    }

    std::vector<std::string> files;
    for (auto& entry : fs::directory_iterator(instances_dir)) {
        if (entry.path().extension() == ".json" &&
            entry.path().filename() != "manifest.json")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "No JSON instances in: " << instances_dir << "\n";
        return 1;
    }

    // TSV header
    std::cout << "Instance\tn\tConfig\tTime_ms\tRC\tMatch_Exact\n";

    for (auto& fpath : files) {
        std::string name;
        std::vector<std::vector<int>> dist;
        std::vector<int> demand;
        int capacity;
        std::vector<double> cover_duals;
        double oracle_rc;

        if (!load_instance(fpath, name, dist, demand, capacity, cover_duals, oracle_rc))
            continue;

        int n = (int)cover_duals.size();

        // Exact RC for comparison (run exact only on n<=18)
        double exact_rc = oracle_rc;
        if (exact_rc > 1e29 && n <= 18) {
            auto [t, rc] = run_exact(dist, demand, capacity, cover_duals, 60000.0);
            exact_rc = rc;
        }

        // ── k-cycle K=2 ──
        {
            auto [t, rc] = run_kcycle<2>(dist, demand, capacity, cover_duals);
            print_row(name, n, "kcycle_K2", t, rc, match(rc, exact_rc));
        }

        // ── k-cycle K=3 ──
        {
            auto [t, rc] = run_kcycle<3>(dist, demand, capacity, cover_duals);
            print_row(name, n, "kcycle_K3", t, rc, match(rc, exact_rc));
        }

        // ── k-cycle K=5 (only n<=18) ──
        if (n <= 18) {
            auto [t, rc] = run_kcycle<5>(dist, demand, capacity, cover_duals, 60000.0);
            print_row(name, n, "kcycle_K5", t, rc, match(rc, exact_rc));
        } else {
            print_row(name, n, "kcycle_K5", 0.0, 1e30, "SKIP");
        }

        // ── ng-route Δ=5 ──
        {
            auto [t, rc] = run_ng(dist, demand, capacity, cover_duals, 5);
            print_row(name, n, "ng_D5", t, rc, match(rc, exact_rc));
        }

        // ── ng-route Δ=8 ──
        {
            auto [t, rc] = run_ng(dist, demand, capacity, cover_duals, 8);
            print_row(name, n, "ng_D8", t, rc, match(rc, exact_rc));
        }

        // ── exact (only n<=18) ──
        if (n <= 18) {
            auto [t, rc] = run_exact(dist, demand, capacity, cover_duals, 60000.0);
            print_row(name, n, "exact", t, rc, "REF");
        } else {
            print_row(name, n, "exact", 0.0, 1e30, "SKIP");
        }

        std::cout.flush();
    }

    return 0;
}
