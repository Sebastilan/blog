// ============================================================================
// bench_optimization.cpp — A/B Benchmark for BIDIR_POOL optimizations
// ============================================================================
// Runs BIDIR_POOL on each instance 3 times, takes the median timing.
// For n <= 15 instances, verifies RC against the exact oracle.
// Outputs a TSV row per instance to stdout.
//
// Usage:
//   bench_optimization [instances_dir]          # JSON instances
//   bench_optimization --stress COUNT [--max-n N]  # random stress instances
//
// Compile-time A/B switches (set via -D flags):
//   -DESPPRC_OPT_REACHABLE_BITMASK=0/1
//   -DESPPRC_OPT_GREEDY_BOUND=0/1
// ============================================================================

#include "relaxed/espprc_ng_bidir_pool.h"
#include "exact/espprc.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>
#include <cmath>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Verify path RC from scratch ──────────────────────────────────────────────
static std::string verify_path_rc(
    const std::vector<int>& path,
    double reported_rc,
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals)
{
    if (path.empty()) return "no_path";
    if (path.front() != 0 || path.back() != 0) return "bad_endpoints";

    int n_cust = (int)cover_duals.size();
    int N = n_cust + 1;

    std::vector<double> c_bar(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c_bar[i * N + j] = (double)dist[i][j];
            if (j >= 1) c_bar[i * N + j] -= cover_duals[j - 1];
        }

    double rc = 0.0;
    int load = 0;
    for (int k = 0; k + 1 < (int)path.size(); k++) {
        int u = path[k], v = path[k + 1];
        if (u < 0 || u >= N || v < 0 || v >= N) return "bad_node";
        rc += c_bar[u * N + v];
        if (v > 0) load += demand[v];
    }

    if (load > capacity) return "capacity_violated";
    if (std::abs(rc - reported_rc) > 0.01)
        return "rc_mismatch(" + std::to_string(rc) + "!=" + std::to_string(reported_rc) + ")";
    return "ok";
}

// ── Run one instance ─────────────────────────────────────────────────────────
struct BenchResult {
    std::string name;
    int n, delta;
    double time_ms;
    int fwd_labels, bwd_labels;
    long long dom_checks, bucket_skipped;
    double rc;
    std::string verified;
};

static BenchResult run_instance(
    const std::string& name,
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    double oracle_rc,    // 1e30 if unknown
    int delta = 8,
    int runs = 3)
{
    int n = (int)cover_duals.size();

    std::vector<espprc_ng_bidir_pool::Result> results(runs);
    std::vector<double> times(runs);

    for (int r = 0; r < runs; r++) {
        results[r] = espprc_ng_bidir_pool::solve(
            dist, demand, capacity, cover_duals, delta,
            /*theta=*/20, /*time_limit_ms=*/60000.0);
        times[r] = results[r].time_ms;
    }

    // Median run
    std::vector<int> idx(runs);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return times[a] < times[b]; });
    auto& res = results[idx[runs / 2]];

    BenchResult br;
    br.name          = name;
    br.n             = n;
    br.delta         = delta;
    br.time_ms       = res.time_ms;
    br.fwd_labels    = res.fwd_labels;
    br.bwd_labels    = res.bwd_labels;
    br.dom_checks    = res.dom_checks;
    br.bucket_skipped = res.bucket_skipped;
    br.rc            = res.optimal_rc;

    // Verify RC
    if (oracle_rc < 1e29) {
        double gap = std::abs(res.optimal_rc - oracle_rc);
        br.verified = (gap < 0.01) ? "ok" : "MISMATCH";
    } else if (n <= 15) {
        // Run exact oracle ourselves
        auto exact = espprc::solve(dist, demand, capacity, cover_duals, 10000.0);
        double gap = std::abs(res.optimal_rc - exact.optimal_rc);
        br.verified = (gap < 0.01) ? "ok" : "MISMATCH";
    } else {
        // Verify the path RC is internally consistent at least
        if (res.optimal_rc < 1e29)
            br.verified = verify_path_rc(res.optimal_path, res.optimal_rc,
                                         dist, demand, capacity, cover_duals);
        else
            br.verified = "no_neg_rc";
    }

    return br;
}

// ── Print TSV header ─────────────────────────────────────────────────────────
static void print_header() {
    std::cout << "name\tn\tdelta\ttime_ms\tfwd_labels\tbwd_labels"
              << "\tdom_checks\tbucket_skipped\trc\tverified\n";
}

// ── Print TSV row ─────────────────────────────────────────────────────────────
static void print_row(const BenchResult& r) {
    std::cout << r.name
              << "\t" << r.n
              << "\t" << r.delta
              << "\t" << std::fixed << std::setprecision(3) << r.time_ms
              << "\t" << r.fwd_labels
              << "\t" << r.bwd_labels
              << "\t" << r.dom_checks
              << "\t" << r.bucket_skipped
              << "\t" << std::fixed << std::setprecision(6) << r.rc
              << "\t" << r.verified
              << "\n";
}

// ── Load JSON instance ───────────────────────────────────────────────────────
static bool load_json_instance(
    const std::string& path,
    std::string& name,
    std::vector<std::vector<int>>& dist,
    std::vector<int>& demand,
    int& capacity,
    std::vector<double>& cover_duals,
    double& oracle_rc)
{
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "Cannot open: " << path << "\n"; return false; }

    json j;
    try { f >> j; } catch (...) { std::cerr << "JSON parse error: " << path << "\n"; return false; }

    name       = j.value("name", path);
    capacity   = j["capacity"].get<int>();
    demand     = j["demand"].get<std::vector<int>>();
    cover_duals = j["cover_duals"].get<std::vector<double>>();

    auto raw_dist = j["dist"].get<std::vector<std::vector<int>>>();
    dist = std::move(raw_dist);

    oracle_rc = 1e30;
    if (j.contains("oracle") && !j["oracle"]["optimal_rc"].is_null())
        oracle_rc = j["oracle"]["optimal_rc"].get<double>();

    return true;
}

// ── Stress test: random instances ────────────────────────────────────────────
static void run_stress(int count, int max_n) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> n_dist(3, max_n);
    std::uniform_int_distribution<int> q_dist(80, 200);
    std::uniform_int_distribution<int> d_dist(5, 30);
    std::uniform_real_distribution<double> pi_dist(-20.0, 50.0);
    std::uniform_int_distribution<int> coord_dist(0, 100);

    print_header();

    for (int t = 0; t < count; t++) {
        int n = n_dist(rng);
        int Q = q_dist(rng);
        int N = n + 1;

        // Generate random 2D Euclidean coordinates
        std::vector<int> x(N), y(N);
        for (int i = 0; i < N; i++) { x[i] = coord_dist(rng); y[i] = coord_dist(rng); }

        std::vector<std::vector<int>> dist(N, std::vector<int>(N, 0));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int dx = x[i] - x[j], dy = y[i] - y[j];
                dist[i][j] = (int)std::round(std::sqrt(dx*dx + dy*dy));
            }

        std::vector<int> demand(N, 0);
        for (int j = 1; j <= n; j++) demand[j] = d_dist(rng);

        std::vector<double> duals(n);
        for (auto& d : duals) d = pi_dist(rng);

        std::string name = "stress_t" + std::to_string(t) + "_n" + std::to_string(n);
        auto br = run_instance(name, dist, demand, Q, duals, 1e30);
        print_row(br);
        std::cout.flush();
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // Print which optimizations are active
    std::cerr << "bench_optimization: "
              << "REACHABLE_BITMASK=" << ESPPRC_OPT_REACHABLE_BITMASK
              << "  GREEDY_BOUND=" << ESPPRC_OPT_GREEDY_BOUND
              << "  BIN_LIMIT=" << ESPPRC_OPT_BIN_LIMIT
              << "  ADAPTIVE_MEET=" << ESPPRC_OPT_ADAPTIVE_MEET
              << "  DIST_SORT=" << ESPPRC_OPT_DIST_SORT
              << "\n";

    bool stress_mode = false;
    int stress_count = 100;
    int max_n = 30;
    std::string instances_dir = "";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--stress") {
            stress_mode = true;
            if (i + 1 < argc) stress_count = std::stoi(argv[++i]);
        } else if (arg == "--max-n") {
            if (i + 1 < argc) max_n = std::stoi(argv[++i]);
        } else {
            instances_dir = arg;
        }
    }

    if (stress_mode) {
        run_stress(stress_count, max_n);
        return 0;
    }

    // JSON instances mode
    if (instances_dir.empty()) {
        // Default: look for testset/instances relative to binary dir
        instances_dir = "testset/instances";
    }

    if (!fs::exists(instances_dir)) {
        std::cerr << "Instances dir not found: " << instances_dir << "\n";
        return 1;
    }

    // Collect JSON files, sorted
    std::vector<std::string> files;
    for (auto& entry : fs::directory_iterator(instances_dir)) {
        if (entry.path().extension() == ".json" &&
            entry.path().filename() != "manifest.json")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "No JSON instances found in: " << instances_dir << "\n";
        return 1;
    }

    print_header();

    int ok_count = 0, mismatch_count = 0;
    for (auto& fpath : files) {
        std::string name;
        std::vector<std::vector<int>> dist;
        std::vector<int> demand;
        int capacity;
        std::vector<double> cover_duals;
        double oracle_rc;

        if (!load_json_instance(fpath, name, dist, demand, capacity, cover_duals, oracle_rc))
            continue;

        auto br = run_instance(name, dist, demand, capacity, cover_duals, oracle_rc);
        print_row(br);
        std::cout.flush();

        if (br.verified == "ok")         ok_count++;
        else if (br.verified.find("MISMATCH") != std::string::npos) mismatch_count++;
    }

    std::cerr << "\nSummary: " << (ok_count + mismatch_count) << " verified, "
              << ok_count << " ok, " << mismatch_count << " MISMATCH\n";
    return (mismatch_count > 0) ? 1 : 0;
}
