// ============================================================================
// ESPPRC Benchmark: Exact vs Bidir vs ng-route vs ng-bidir vs ng+CB vs Bucket
// ============================================================================
// Runs all solvers, compares RC/timing, and rigorously verifies paths.
//
// Usage:
//   bench_espprc [instances_dir] [--up-to N]       # JSON instances
//   bench_espprc --stress COUNT [--max-n N]         # random stress test
// ============================================================================

#include "exact/espprc.h"
#include "_archive/espprc_bidir.h"
#include "relaxed/espprc_ng.h"
#include "_archive/espprc_ng_bidir.h"
#include "_archive/espprc_ng_cb.h"
#include "_archive/espprc_ng_bucket.h"
#include "_archive/espprc_ng_dssr.h"
#include "_archive/espprc_ng_bidir_par.h"
#include "_archive/espprc_ng_rcsort.h"
#include "_archive/espprc_ng_bucket2.h"
#include "_archive/espprc_ng_pool.h"
#include "exact/espprc_dssr_pool.h"
#include "exact/espprc_exact_pool.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>
#include <iomanip>
#include <random>
#include <set>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Independent path verification ────────────────────────────────────────────
// Recomputes RC from scratch using the path and c_bar matrix.
// Returns error string (empty = OK).
static std::string verify_path(
    const std::vector<int>& path,
    double reported_rc,
    const std::vector<std::vector<int>>& dist,
    const std::vector<int>& demand,
    int capacity,
    const std::vector<double>& cover_duals,
    bool check_elementary = true)
{
    if (path.empty()) return "empty path";
    if (path.front() != 0) return "path doesn't start at depot";
    if (path.back() != 0)  return "path doesn't end at depot";
    if (path.size() < 3)   return "path too short (no customers)";

    int n_cust = (int)cover_duals.size();

    // Check elementary (optional — ng paths may repeat)
    if (check_elementary) {
        std::set<int> seen;
        for (int v : path) {
            if (v == 0) continue;
            if (v < 1 || v > n_cust) return "invalid node " + std::to_string(v);
            if (seen.count(v)) return "repeated node " + std::to_string(v);
            seen.insert(v);
        }
    }

    // Check capacity
    int load = 0;
    for (int v : path)
        if (v != 0) load += demand[v];
    if (load > capacity)
        return "capacity violated: " + std::to_string(load) + " > " + std::to_string(capacity);

    // Independently recompute RC from c_bar
    double recomputed_rc = 0.0;
    for (int k = 0; k + 1 < (int)path.size(); k++) {
        int i = path[k], j = path[k + 1];
        double edge_cost = (double)dist[i][j];
        if (j >= 1) edge_cost -= cover_duals[j - 1];
        recomputed_rc += edge_cost;
    }

    double gap = std::abs(recomputed_rc - reported_rc);
    if (gap > 1e-6)
        return "RC mismatch: path gives " + std::to_string(recomputed_rc)
             + " but reported " + std::to_string(reported_rc);

    return "";  // OK
}

// ── JSON instance loader ─────────────────────────────────────────────────────
struct TestInstance {
    std::string name;
    int n_customers;
    std::vector<std::vector<int>> dist;
    std::vector<int> demand;
    int capacity;
    std::vector<double> cover_duals;
    double oracle_rc;
    std::string oracle_method;
};

TestInstance load_instance(const fs::path& filepath) {
    std::ifstream ifs(filepath);
    json j = json::parse(ifs);

    TestInstance inst;
    inst.name        = j["name"];
    inst.n_customers = j["n_customers"];
    inst.capacity    = j["capacity"];

    auto& jd = j["dist"];
    int N = (int)jd.size();
    inst.dist.resize(N, std::vector<int>(N));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            inst.dist[i][k] = jd[i][k].get<int>();

    for (auto& v : j["demand"])
        inst.demand.push_back(v.get<int>());
    for (auto& v : j["cover_duals"])
        inst.cover_duals.push_back(v.get<double>());

    auto& oracle = j["oracle"];
    inst.oracle_rc = oracle["optimal_rc"].is_null() ? 1e30 : oracle["optimal_rc"].get<double>();
    inst.oracle_method = oracle["method"].get<std::string>();

    return inst;
}

// ── Benchmark on JSON instances ──────────────────────────────────────────────
int run_json_benchmark(const std::string& inst_dir, int max_n) {
    double time_limit = 60000.0;

    std::vector<fs::path> files;
    for (auto& entry : fs::directory_iterator(inst_dir)) {
        auto fn = entry.path().filename().string();
        if (fn.size() > 5 && fn.substr(fn.size() - 5) == ".json"
            && fn != "manifest.json")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    int errors = 0;

    // ── Part 1: Exact vs Bidir (only n <= 15) ───────────────────────────────
    std::cout << "=== Part 1: Exact vs Bidirectional (n <= 15) ===\n";
    std::cout << std::string(105, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "Oracle RC"
              << std::setw(11) << "Uni RC"
              << std::setw(9)  << "Uni(ms)"
              << std::setw(9)  << "UniLbl"
              << std::setw(11) << "Bidir RC"
              << std::setw(9)  << "Bid(ms)"
              << std::setw(9)  << "BidLbl"
              << "  Speedup  Verify\n";
    std::cout << std::string(105, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n || inst.n_customers > 15) continue;

        espprc::Result uni = espprc::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, time_limit);
        espprc_bidir::Result bid = espprc_bidir::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, time_limit);

        char buf_oracle[16], buf_uni[16], buf_bid[16];
        if (inst.oracle_rc > 1e20) snprintf(buf_oracle, 16, "N/A");
        else snprintf(buf_oracle, 16, "%.2f", inst.oracle_rc);

        if (uni.timed_out) snprintf(buf_uni, 16, "T/O");
        else if (uni.optimal_rc > 1e20) snprintf(buf_uni, 16, "none");
        else snprintf(buf_uni, 16, "%.2f", uni.optimal_rc);

        if (bid.timed_out) snprintf(buf_bid, 16, "T/O");
        else if (bid.optimal_rc > 1e20) snprintf(buf_bid, 16, "none");
        else snprintf(buf_bid, 16, "%.2f", bid.optimal_rc);

        char buf_speedup[16];
        if (uni.timed_out && bid.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (uni.timed_out)
            snprintf(buf_speedup, 16, ">%.0fx", time_limit / bid.time_ms);
        else if (bid.timed_out)
            snprintf(buf_speedup, 16, "<1x");
        else if (bid.time_ms > 0.001)
            snprintf(buf_speedup, 16, "%.1fx", uni.time_ms / bid.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        std::string verify_status = "OK";

        if (!uni.timed_out && uni.optimal_rc < 1e20) {
            std::string err = verify_path(uni.optimal_path, uni.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, true);
            if (!err.empty()) { verify_status = "UNI:" + err; errors++; }
        }

        if (!bid.timed_out && bid.optimal_rc < 1e20) {
            std::string err = verify_path(bid.optimal_path, bid.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, true);
            if (!err.empty()) { verify_status = "BID:" + err; errors++; }
        }

        if (!uni.timed_out && !bid.timed_out) {
            if (uni.optimal_rc < 1e20 && bid.optimal_rc < 1e20) {
                if (std::abs(uni.optimal_rc - bid.optimal_rc) > 0.01) {
                    verify_status = "MISMATCH";
                    errors++;
                }
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_oracle
                  << std::setw(11) << buf_uni
                  << std::setw(9)  << std::fixed << std::setprecision(1) << uni.time_ms
                  << std::setw(9)  << uni.labels_created
                  << std::setw(11) << buf_bid
                  << std::setw(9)  << std::fixed << std::setprecision(1) << bid.time_ms
                  << std::setw(9)  << (bid.fwd_labels + bid.bwd_labels)
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << verify_status << std::endl;
    }
    std::cout << std::string(105, '-') << "\n\n";

    // ── Part 2: ng-route uni vs bidir (all instances, delta=8) ──────────────
    std::cout << "=== Part 2: ng-route Unidirectional vs Bidirectional (delta=8) ===\n";
    std::cout << std::string(120, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "Oracle RC"
              << std::setw(11) << "ng_uni RC"
              << std::setw(9)  << "uni(ms)"
              << std::setw(9)  << "uniLbl"
              << std::setw(13) << "ng_bidir RC"
              << std::setw(9)  << "bid(ms)"
              << std::setw(9)  << "bidLbl"
              << "  Speedup  Elem?  Verify\n";
    std::cout << std::string(120, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        espprc_ng::Result ng_uni = espprc_ng::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        espprc_ng_bidir::Result ng_bid = espprc_ng_bidir::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        char buf_oracle[16], buf_ngu[16], buf_ngb[16];
        if (inst.oracle_rc > 1e20) snprintf(buf_oracle, 16, "N/A");
        else snprintf(buf_oracle, 16, "%.2f", inst.oracle_rc);

        if (ng_uni.timed_out) snprintf(buf_ngu, 16, "T/O");
        else if (ng_uni.optimal_rc > 1e20) snprintf(buf_ngu, 16, "none");
        else snprintf(buf_ngu, 16, "%.2f", ng_uni.optimal_rc);

        if (ng_bid.timed_out) snprintf(buf_ngb, 16, "T/O");
        else if (ng_bid.optimal_rc > 1e20) snprintf(buf_ngb, 16, "none");
        else snprintf(buf_ngb, 16, "%.2f", ng_bid.optimal_rc);

        char buf_speedup[16];
        if (ng_uni.timed_out && ng_bid.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (ng_uni.timed_out)
            snprintf(buf_speedup, 16, ">%.0fx", time_limit / ng_bid.time_ms);
        else if (ng_bid.timed_out)
            snprintf(buf_speedup, 16, "<1x");
        else if (ng_bid.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", ng_uni.time_ms / ng_bid.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        std::string verify_status = "OK";

        // Verify ng_uni path
        if (!ng_uni.timed_out && ng_uni.optimal_rc < 1e20) {
            std::string err = verify_path(ng_uni.optimal_path, ng_uni.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, false);
            if (!err.empty()) { verify_status = "NGU:" + err; errors++; }
        }

        // Verify ng_bidir path
        if (!ng_bid.timed_out && ng_bid.optimal_rc < 1e20) {
            std::string err = verify_path(ng_bid.optimal_path, ng_bid.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, false);
            if (!err.empty()) { verify_status = "NGB:" + err; errors++; }
        }

        // Both are relaxations: RC should be <= exact oracle RC
        // (But ng_bidir and ng_uni may differ from each other)

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_oracle
                  << std::setw(11) << buf_ngu
                  << std::setw(9)  << std::fixed << std::setprecision(1) << ng_uni.time_ms
                  << std::setw(9)  << ng_uni.labels_created
                  << std::setw(13) << buf_ngb
                  << std::setw(9)  << std::fixed << std::setprecision(1) << ng_bid.time_ms
                  << std::setw(9)  << (ng_bid.fwd_labels + ng_bid.bwd_labels)
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << std::setw(6) << (ng_uni.path_is_elementary ? "yes" : "no")
                  << " " << verify_status << std::endl;
    }
    std::cout << std::string(120, '-') << "\n\n";

    // ── Part 3: ng-route vs ng+Completion Bounds (all instances, delta=8) ───
    std::cout << "=== Part 3: ng-route vs ng+Completion Bounds (delta=8) ===\n";
    std::cout << std::string(120, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "ng RC"
              << std::setw(9)  << "ng(ms)"
              << std::setw(9)  << "ngLbl"
              << std::setw(11) << "ng+CB RC"
              << std::setw(9)  << "CB(ms)"
              << std::setw(9)  << "CBLbl"
              << std::setw(9)  << "Pruned"
              << "  Speedup  Prune%  Verify\n";
    std::cout << std::string(120, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        espprc_ng::Result ng = espprc_ng::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        espprc_ng_cb::Result cb = espprc_ng_cb::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        char buf_ng[16], buf_cb[16];
        if (ng.timed_out) snprintf(buf_ng, 16, "T/O");
        else if (ng.optimal_rc > 1e20) snprintf(buf_ng, 16, "none");
        else snprintf(buf_ng, 16, "%.2f", ng.optimal_rc);

        if (cb.timed_out) snprintf(buf_cb, 16, "T/O");
        else if (cb.optimal_rc > 1e20) snprintf(buf_cb, 16, "none");
        else snprintf(buf_cb, 16, "%.2f", cb.optimal_rc);

        char buf_speedup[16];
        if (ng.timed_out && cb.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (cb.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", ng.time_ms / cb.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        // Prune percentage: pruned / (pruned + labels_created)
        char buf_prune[16];
        int total_attempts = cb.labels_pruned + cb.labels_created;
        if (total_attempts > 0)
            snprintf(buf_prune, 16, "%.0f%%", 100.0 * cb.labels_pruned / total_attempts);
        else
            snprintf(buf_prune, 16, "0%%");

        std::string verify_status = "OK";

        // Verify ng+CB path
        if (!cb.timed_out && cb.optimal_rc < 1e20) {
            std::string err = verify_path(cb.optimal_path, cb.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, false);
            if (!err.empty()) { verify_status = "CB:" + err; errors++; }
        }

        // When ng finds negative RC, cb must agree (pure pruning).
        // When both RC >= 0, mismatch is benign (no negative RC column).
        if (!ng.timed_out && !cb.timed_out
            && ng.optimal_rc < 1e20 && cb.optimal_rc < 1e20) {
            bool ng_negative = ng.optimal_rc < -1e-6;
            if (ng_negative && std::abs(ng.optimal_rc - cb.optimal_rc) > 0.01) {
                verify_status = "MISMATCH";
                errors++;
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_ng
                  << std::setw(9)  << std::fixed << std::setprecision(1) << ng.time_ms
                  << std::setw(9)  << ng.labels_created
                  << std::setw(11) << buf_cb
                  << std::setw(9)  << std::fixed << std::setprecision(1) << cb.time_ms
                  << std::setw(9)  << cb.labels_created
                  << std::setw(9)  << cb.labels_pruned
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << std::setw(8) << buf_prune
                  << verify_status << std::endl;
    }
    std::cout << std::string(120, '-') << "\n\n";

    // ── Part 4: ng-route vs ng+Bucket Graph (all instances, delta=8) ─────────
    std::cout << "=== Part 4: ng-route vs ng+Bucket Graph (delta=8, theta=20) ===\n";
    std::cout << std::string(120, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "ng RC"
              << std::setw(9)  << "ng(ms)"
              << std::setw(9)  << "ngLbl"
              << std::setw(11) << "bkt RC"
              << std::setw(9)  << "bkt(ms)"
              << std::setw(9)  << "bktLbl"
              << std::setw(9)  << "domChk"
              << "  Speedup  Verify\n";
    std::cout << std::string(120, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        espprc_ng::Result ng = espprc_ng::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        espprc_ng_bucket::Result bkt = espprc_ng_bucket::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, 20, time_limit);

        char buf_ng[16], buf_bkt[16];
        if (ng.timed_out) snprintf(buf_ng, 16, "T/O");
        else if (ng.optimal_rc > 1e20) snprintf(buf_ng, 16, "none");
        else snprintf(buf_ng, 16, "%.2f", ng.optimal_rc);

        if (bkt.timed_out) snprintf(buf_bkt, 16, "T/O");
        else if (bkt.optimal_rc > 1e20) snprintf(buf_bkt, 16, "none");
        else snprintf(buf_bkt, 16, "%.2f", bkt.optimal_rc);

        char buf_speedup[16];
        if (ng.timed_out && bkt.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (bkt.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", ng.time_ms / bkt.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        // Format dom_checks (K or M)
        char buf_dom[16];
        if (bkt.dom_checks > 1000000)
            snprintf(buf_dom, 16, "%.1fM", bkt.dom_checks / 1e6);
        else if (bkt.dom_checks > 1000)
            snprintf(buf_dom, 16, "%.0fK", bkt.dom_checks / 1e3);
        else
            snprintf(buf_dom, 16, "%d", bkt.dom_checks);

        std::string verify_status = "OK";

        // Verify bucket path
        if (!bkt.timed_out && bkt.optimal_rc < 1e20) {
            std::string err = verify_path(bkt.optimal_path, bkt.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, false);
            if (!err.empty()) { verify_status = "BKT:" + err; errors++; }
        }

        // RC must match ng (same algorithm, just different data structure)
        if (!ng.timed_out && !bkt.timed_out
            && ng.optimal_rc < 1e20 && bkt.optimal_rc < 1e20) {
            if (std::abs(ng.optimal_rc - bkt.optimal_rc) > 0.01) {
                verify_status = "MISMATCH";
                errors++;
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_ng
                  << std::setw(9)  << std::fixed << std::setprecision(1) << ng.time_ms
                  << std::setw(9)  << ng.labels_created
                  << std::setw(11) << buf_bkt
                  << std::setw(9)  << std::fixed << std::setprecision(1) << bkt.time_ms
                  << std::setw(9)  << bkt.labels_created
                  << std::setw(9)  << buf_dom
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << verify_status << std::endl;
    }
    std::cout << std::string(120, '-') << "\n\n";

    // ── Part 5: ng-route vs ng+DSSR (all instances, delta=8) ─────────────────
    std::cout << "=== Part 5: ng-route vs ng+DSSR (delta=8) ===\n";
    std::cout << std::string(130, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "ng RC"
              << std::setw(9)  << "ng(ms)"
              << std::setw(9)  << "ngLbl"
              << std::setw(6)  << "elem"
              << std::setw(11) << "dssr RC"
              << std::setw(9)  << "dssr(ms)"
              << std::setw(9)  << "dsLbl"
              << std::setw(6)  << "elem"
              << std::setw(5)  << "iter"
              << std::setw(5)  << "crit"
              << "  Speedup  Verify\n";
    std::cout << std::string(130, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        espprc_ng::Result ng = espprc_ng::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        espprc_ng_dssr::Result dssr = espprc_ng_dssr::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, 20, time_limit);

        char buf_ng[16], buf_dssr[16];
        if (ng.timed_out) snprintf(buf_ng, 16, "T/O");
        else if (ng.optimal_rc > 1e20) snprintf(buf_ng, 16, "none");
        else snprintf(buf_ng, 16, "%.2f", ng.optimal_rc);

        if (dssr.timed_out) snprintf(buf_dssr, 16, "T/O");
        else if (dssr.optimal_rc > 1e20) snprintf(buf_dssr, 16, "none");
        else snprintf(buf_dssr, 16, "%.2f", dssr.optimal_rc);

        char buf_speedup[16];
        if (ng.timed_out && dssr.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (dssr.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", ng.time_ms / dssr.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        std::string verify_status = "OK";

        // Verify DSSR path
        if (!dssr.timed_out && dssr.optimal_rc < 1e20) {
            std::string err = verify_path(dssr.optimal_path, dssr.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, false);
            if (!err.empty()) { verify_status = "DSSR:" + err; errors++; }
        }

        // DSSR is tighter than ng: dssr_rc >= ng_rc (less relaxed)
        if (!ng.timed_out && !dssr.timed_out
            && ng.optimal_rc < 1e20 && dssr.optimal_rc < 1e20) {
            if (dssr.optimal_rc < ng.optimal_rc - 0.01) {
                verify_status = "DSSR_TIGHTER_WRONG";
                errors++;
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_ng
                  << std::setw(9)  << std::fixed << std::setprecision(1) << ng.time_ms
                  << std::setw(9)  << ng.labels_created
                  << std::setw(6)  << (ng.path_is_elementary ? "Y" : "N")
                  << std::setw(11) << buf_dssr
                  << std::setw(9)  << std::fixed << std::setprecision(1) << dssr.time_ms
                  << std::setw(9)  << dssr.labels_created
                  << std::setw(6)  << (dssr.path_is_elementary ? "Y" : "N")
                  << std::setw(5)  << dssr.dssr_iterations
                  << std::setw(5)  << dssr.critical_count
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << verify_status << std::endl;
    }
    std::cout << std::string(130, '-') << "\n\n";

    // ── Part 6: ng-bidir sequential vs parallel (delta=8) ────────────────────
    std::cout << "=== Part 6: ng-bidir Sequential vs Parallel (delta=8) ===\n";
    std::cout << std::string(120, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "seq RC"
              << std::setw(9)  << "seq(ms)"
              << std::setw(9)  << "seqLbl"
              << std::setw(11) << "par RC"
              << std::setw(9)  << "par(ms)"
              << std::setw(9)  << "parLbl"
              << "  Speedup  Verify\n";
    std::cout << std::string(120, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        espprc_ng_bidir::Result seq = espprc_ng_bidir::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        espprc_ng_bidir_par::Result par = espprc_ng_bidir_par::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        char buf_seq[16], buf_par[16];
        if (seq.timed_out) snprintf(buf_seq, 16, "T/O");
        else if (seq.optimal_rc > 1e20) snprintf(buf_seq, 16, "none");
        else snprintf(buf_seq, 16, "%.2f", seq.optimal_rc);

        if (par.timed_out) snprintf(buf_par, 16, "T/O");
        else if (par.optimal_rc > 1e20) snprintf(buf_par, 16, "none");
        else snprintf(buf_par, 16, "%.2f", par.optimal_rc);

        char buf_speedup[16];
        if (seq.timed_out && par.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (par.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", seq.time_ms / par.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        std::string verify_status = "OK";

        if (!par.timed_out && par.optimal_rc < 1e20) {
            std::string err = verify_path(par.optimal_path, par.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, false);
            if (!err.empty()) { verify_status = "PAR:" + err; errors++; }
        }

        // RC must match sequential
        if (!seq.timed_out && !par.timed_out
            && seq.optimal_rc < 1e20 && par.optimal_rc < 1e20) {
            if (std::abs(seq.optimal_rc - par.optimal_rc) > 0.01) {
                verify_status = "MISMATCH";
                errors++;
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_seq
                  << std::setw(9)  << std::fixed << std::setprecision(1) << seq.time_ms
                  << std::setw(9)  << (seq.fwd_labels + seq.bwd_labels)
                  << std::setw(11) << buf_par
                  << std::setw(9)  << std::fixed << std::setprecision(1) << par.time_ms
                  << std::setw(9)  << (par.fwd_labels + par.bwd_labels)
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << verify_status << std::endl;
    }
    std::cout << std::string(120, '-') << "\n\n";

    // ── Part 7: ng-route vs ng+RC-Sorted Dominance (all instances, delta=8) ──
    std::cout << "=== Part 7: ng-route vs ng+RC-Sorted Dominance (delta=8) ===\n";
    std::cout << std::string(130, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "ng RC"
              << std::setw(9)  << "ng(ms)"
              << std::setw(9)  << "ngLbl"
              << std::setw(11) << "rcs RC"
              << std::setw(9)  << "rcs(ms)"
              << std::setw(9)  << "rcsLbl"
              << std::setw(9)  << "domChk"
              << std::setw(9)  << "skipped"
              << "  Speedup  Verify\n";
    std::cout << std::string(130, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        espprc_ng::Result ng = espprc_ng::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        espprc_ng_rcsort::Result rcs = espprc_ng_rcsort::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        char buf_ng[16], buf_rcs[16];
        if (ng.timed_out) snprintf(buf_ng, 16, "T/O");
        else if (ng.optimal_rc > 1e20) snprintf(buf_ng, 16, "none");
        else snprintf(buf_ng, 16, "%.2f", ng.optimal_rc);

        if (rcs.timed_out) snprintf(buf_rcs, 16, "T/O");
        else if (rcs.optimal_rc > 1e20) snprintf(buf_rcs, 16, "none");
        else snprintf(buf_rcs, 16, "%.2f", rcs.optimal_rc);

        char buf_speedup[16];
        if (ng.timed_out && rcs.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (rcs.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", ng.time_ms / rcs.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        char buf_dom[16], buf_skip[16];
        if (rcs.dom_checks > 1000000)
            snprintf(buf_dom, 16, "%.1fM", rcs.dom_checks / 1e6);
        else if (rcs.dom_checks > 1000)
            snprintf(buf_dom, 16, "%.0fK", rcs.dom_checks / 1e3);
        else
            snprintf(buf_dom, 16, "%lld", rcs.dom_checks);

        if (rcs.dom_skipped > 1000000)
            snprintf(buf_skip, 16, "%.1fM", rcs.dom_skipped / 1e6);
        else if (rcs.dom_skipped > 1000)
            snprintf(buf_skip, 16, "%.0fK", rcs.dom_skipped / 1e3);
        else
            snprintf(buf_skip, 16, "%lld", rcs.dom_skipped);

        std::string verify_status = "OK";

        if (!rcs.timed_out && rcs.optimal_rc < 1e20) {
            std::string err = verify_path(rcs.optimal_path, rcs.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, false);
            if (!err.empty()) { verify_status = "RCS:" + err; errors++; }
        }

        // RC must match ng (same algorithm, just different dominance order)
        if (!ng.timed_out && !rcs.timed_out
            && ng.optimal_rc < 1e20 && rcs.optimal_rc < 1e20) {
            if (std::abs(ng.optimal_rc - rcs.optimal_rc) > 0.01) {
                verify_status = "MISMATCH";
                errors++;
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_ng
                  << std::setw(9)  << std::fixed << std::setprecision(1) << ng.time_ms
                  << std::setw(9)  << ng.labels_created
                  << std::setw(11) << buf_rcs
                  << std::setw(9)  << std::fixed << std::setprecision(1) << rcs.time_ms
                  << std::setw(9)  << rcs.labels_created
                  << std::setw(9)  << buf_dom
                  << std::setw(9)  << buf_skip
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << verify_status << std::endl;
    }
    std::cout << std::string(130, '-') << "\n\n";

    // ── Part 8: ng+RC-Sort vs ng+BucketV2 (RC-sorted bins) ──────────────────
    std::cout << "=== Part 8: ng+RC-Sort vs ng+Bucket v2 (RC-sorted bins, theta=20) ===\n";
    std::cout << std::string(140, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "rcs RC"
              << std::setw(9)  << "rcs(ms)"
              << std::setw(9)  << "rcsLbl"
              << std::setw(11) << "bk2 RC"
              << std::setw(9)  << "bk2(ms)"
              << std::setw(9)  << "bk2Lbl"
              << std::setw(9)  << "domChk"
              << std::setw(9)  << "rcSkip"
              << std::setw(9)  << "bktSkip"
              << "  Speedup  Verify\n";
    std::cout << std::string(140, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        espprc_ng_rcsort::Result rcs = espprc_ng_rcsort::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        espprc_ng_bucket2::Result bk2 = espprc_ng_bucket2::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, 20, time_limit);

        char buf_rcs[16], buf_bk2[16];
        if (rcs.timed_out) snprintf(buf_rcs, 16, "T/O");
        else if (rcs.optimal_rc > 1e20) snprintf(buf_rcs, 16, "none");
        else snprintf(buf_rcs, 16, "%.2f", rcs.optimal_rc);

        if (bk2.timed_out) snprintf(buf_bk2, 16, "T/O");
        else if (bk2.optimal_rc > 1e20) snprintf(buf_bk2, 16, "none");
        else snprintf(buf_bk2, 16, "%.2f", bk2.optimal_rc);

        char buf_speedup[16];
        if (rcs.timed_out && bk2.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (bk2.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", rcs.time_ms / bk2.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        auto fmt_count = [](char* buf, long long val) {
            if (val > 1000000) snprintf(buf, 16, "%.1fM", val / 1e6);
            else if (val > 1000) snprintf(buf, 16, "%.0fK", val / 1e3);
            else snprintf(buf, 16, "%lld", val);
        };
        char buf_dom[16], buf_rcskip[16], buf_bktskip[16];
        fmt_count(buf_dom, bk2.dom_checks);
        fmt_count(buf_rcskip, bk2.dom_skipped);
        fmt_count(buf_bktskip, bk2.bucket_skipped);

        std::string verify_status = "OK";

        if (!bk2.timed_out && bk2.optimal_rc < 1e20) {
            std::string err = verify_path(bk2.optimal_path, bk2.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, false);
            if (!err.empty()) { verify_status = "BK2:" + err; errors++; }
        }

        // RC must match rcsort (same ng algorithm)
        if (!rcs.timed_out && !bk2.timed_out
            && rcs.optimal_rc < 1e20 && bk2.optimal_rc < 1e20) {
            if (std::abs(rcs.optimal_rc - bk2.optimal_rc) > 0.01) {
                verify_status = "MISMATCH";
                errors++;
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_rcs
                  << std::setw(9)  << std::fixed << std::setprecision(1) << rcs.time_ms
                  << std::setw(9)  << rcs.labels_created
                  << std::setw(11) << buf_bk2
                  << std::setw(9)  << std::fixed << std::setprecision(1) << bk2.time_ms
                  << std::setw(9)  << bk2.labels_created
                  << std::setw(9)  << buf_dom
                  << std::setw(9)  << buf_rcskip
                  << std::setw(9)  << buf_bktskip
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << verify_status << std::endl;
    }
    std::cout << std::string(140, '-') << "\n\n";

    // ── Part 9: Bucket v2 vs Pool (pre-allocated) ─────────────────────────────
    std::cout << "=== Part 9: Bucket v2 vs Pool (pre-allocated labels+buckets) ===\n";
    std::cout << std::string(110, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "bk2 RC"
              << std::setw(9)  << "bk2(ms)"
              << std::setw(9)  << "bk2Lbl"
              << std::setw(11) << "pool RC"
              << std::setw(9)  << "pool(ms)"
              << std::setw(9)  << "poolLbl"
              << "  Speedup  Verify\n";
    std::cout << std::string(110, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        espprc_ng_bucket2::Result bk2 = espprc_ng_bucket2::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, 20, time_limit);

        espprc_ng_pool::Result pool = espprc_ng_pool::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, 20, time_limit);

        char buf_bk2[16], buf_pool[16];
        if (bk2.timed_out) snprintf(buf_bk2, 16, "T/O");
        else if (bk2.optimal_rc > 1e20) snprintf(buf_bk2, 16, "none");
        else snprintf(buf_bk2, 16, "%.2f", bk2.optimal_rc);

        if (pool.timed_out) snprintf(buf_pool, 16, "T/O");
        else if (pool.optimal_rc > 1e20) snprintf(buf_pool, 16, "none");
        else snprintf(buf_pool, 16, "%.2f", pool.optimal_rc);

        char buf_speedup[16];
        if (bk2.timed_out && pool.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (pool.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", bk2.time_ms / pool.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        std::string verify_status = "OK";

        if (!pool.timed_out && pool.optimal_rc < 1e20) {
            std::string err = verify_path(pool.optimal_path, pool.optimal_rc,
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, false);
            if (!err.empty()) { verify_status = "POOL:" + err; errors++; }
        }

        if (!bk2.timed_out && !pool.timed_out
            && bk2.optimal_rc < 1e20 && pool.optimal_rc < 1e20) {
            if (std::abs(bk2.optimal_rc - pool.optimal_rc) > 0.01) {
                verify_status = "MISMATCH";
                errors++;
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_bk2
                  << std::setw(9)  << std::fixed << std::setprecision(1) << bk2.time_ms
                  << std::setw(9)  << bk2.labels_created
                  << std::setw(11) << buf_pool
                  << std::setw(9)  << std::fixed << std::setprecision(1) << pool.time_ms
                  << std::setw(9)  << pool.labels_created
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << verify_status << std::endl;
    }
    std::cout << std::string(110, '-') << "\n\n";

    // ── Part 10: End-to-End: ng (baseline) vs Pool (best) ────────────────────
    std::cout << "=== Part 10: End-to-End  ng (baseline) vs Pool (best) ===\n";
    std::cout << std::string(120, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "ng RC"
              << std::setw(9)  << "ng(ms)"
              << std::setw(9)  << "ngLbl"
              << std::setw(11) << "pool RC"
              << std::setw(9)  << "pool(ms)"
              << std::setw(9)  << "poolLbl"
              << "  Speedup  Verify\n";
    std::cout << std::string(120, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        espprc_ng::Result ng = espprc_ng::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, time_limit);

        espprc_ng_pool::Result pool = espprc_ng_pool::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, 20, time_limit);

        char buf_ng[16], buf_pool[16];
        if (ng.timed_out) snprintf(buf_ng, 16, "T/O");
        else if (ng.optimal_rc > 1e20) snprintf(buf_ng, 16, "none");
        else snprintf(buf_ng, 16, "%.2f", ng.optimal_rc);

        if (pool.timed_out) snprintf(buf_pool, 16, "T/O");
        else if (pool.optimal_rc > 1e20) snprintf(buf_pool, 16, "none");
        else snprintf(buf_pool, 16, "%.2f", pool.optimal_rc);

        char buf_speedup[16];
        if (ng.timed_out && pool.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (pool.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", ng.time_ms / pool.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        std::string verify_status = "OK";
        if (!ng.timed_out && !pool.timed_out
            && ng.optimal_rc < 1e20 && pool.optimal_rc < 1e20) {
            if (std::abs(ng.optimal_rc - pool.optimal_rc) > 0.01) {
                verify_status = "MISMATCH";
                errors++;
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_ng
                  << std::setw(9)  << std::fixed << std::setprecision(1) << ng.time_ms
                  << std::setw(9)  << ng.labels_created
                  << std::setw(11) << buf_pool
                  << std::setw(9)  << std::fixed << std::setprecision(1) << pool.time_ms
                  << std::setw(9)  << pool.labels_created
                  << "  " << std::left << std::setw(7) << buf_speedup
                  << verify_status << std::endl;
    }
    std::cout << std::string(120, '-') << "\n\n";

    // ── Part 11: L3 Exact Pricing: DSSR+Pool vs FullBitmask+Pool ─────────
    std::cout << "=== Part 11: L3 Exact — DSSR+Pool vs FullBitmask+Pool ===\n";
    std::cout << std::string(150, '=') << "\n";
    std::cout << std::left  << std::setw(22) << "Instance"
              << std::right << std::setw(3)  << "n"
              << std::setw(11) << "oracle RC"
              << std::setw(11) << "dssr RC"
              << std::setw(9)  << "dssr(ms)"
              << std::setw(9)  << "dsrLbl"
              << std::setw(5)  << "iter"
              << std::setw(5)  << "crit"
              << std::setw(5)  << "elem"
              << std::setw(11) << "exact RC"
              << std::setw(9)  << "ext(ms)"
              << std::setw(9)  << "extLbl"
              << std::setw(5)  << "elem"
              << std::setw(11) << "pool RC"
              << std::setw(9)  << "pool(ms)"
              << "  dssr/ext  Verify\n";
    std::cout << std::string(150, '-') << "\n";

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);
        if (inst.n_customers > max_n) continue;

        int delta = 8;

        // Oracle: exact bitmask (only for n <= 15 where it's reliable)
        espprc::Result oracle;
        if (inst.n_customers <= 15) {
            oracle = espprc::solve(
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, time_limit);
        } else {
            oracle.optimal_rc = 1e30;
            oracle.timed_out = true;
        }

        // DSSR + Pool
        espprc_dssr_pool::Result dssr = espprc_dssr_pool::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, 20, 20, time_limit);

        // Full Bitmask + Pool (only for n <= 32)
        espprc_exact_pool::Result extp;
        if (inst.n_customers <= 32) {
            extp = espprc_exact_pool::solve(
                inst.dist, inst.demand, inst.capacity, inst.cover_duals, 20, time_limit);
        } else {
            extp.optimal_rc = 1e30;
            extp.timed_out = true;
            extp.time_ms = 0;
            extp.labels_created = 0;
            extp.path_is_elementary = true;
        }

        // ng Pool (L2 reference)
        espprc_ng_pool::Result pool = espprc_ng_pool::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, delta, 20, time_limit);

        char buf_orc[16], buf_dssr[16], buf_extp[16], buf_pool[16];
        if (oracle.timed_out) snprintf(buf_orc, 16, "T/O");
        else if (oracle.optimal_rc > 1e20) snprintf(buf_orc, 16, "none");
        else snprintf(buf_orc, 16, "%.2f", oracle.optimal_rc);

        if (dssr.timed_out) snprintf(buf_dssr, 16, "T/O");
        else if (dssr.optimal_rc > 1e20) snprintf(buf_dssr, 16, "none");
        else snprintf(buf_dssr, 16, "%.2f", dssr.optimal_rc);

        if (extp.timed_out) snprintf(buf_extp, 16, "T/O");
        else if (extp.optimal_rc > 1e20) snprintf(buf_extp, 16, "none");
        else snprintf(buf_extp, 16, "%.2f", extp.optimal_rc);

        if (pool.timed_out) snprintf(buf_pool, 16, "T/O");
        else if (pool.optimal_rc > 1e20) snprintf(buf_pool, 16, "none");
        else snprintf(buf_pool, 16, "%.2f", pool.optimal_rc);

        char buf_speedup[16];
        if (dssr.timed_out && extp.timed_out)
            snprintf(buf_speedup, 16, "---");
        else if (extp.time_ms > 0.01)
            snprintf(buf_speedup, 16, "%.1fx", extp.time_ms / dssr.time_ms);
        else
            snprintf(buf_speedup, 16, "~");

        // Verify: both exact solvers should match oracle (only when oracle didn't time out)
        std::string verify_status = "OK";
        if (!oracle.timed_out && !dssr.timed_out
            && dssr.optimal_rc < 1e20 && oracle.optimal_rc < 1e20) {
            if (std::abs(dssr.optimal_rc - oracle.optimal_rc) > 0.01
                && dssr.path_is_elementary) {
                verify_status = "DSSR_MISMATCH";
                errors++;
            }
        }
        if (!oracle.timed_out && !extp.timed_out
            && extp.optimal_rc < 1e20 && oracle.optimal_rc < 1e20) {
            if (std::abs(extp.optimal_rc - oracle.optimal_rc) > 0.01) {
                verify_status = "EXACT_MISMATCH";
                errors++;
            }
        }
        // Cross-check: when both L3 solvers finish and produce elementary paths,
        // they should agree
        if (!dssr.timed_out && !extp.timed_out
            && dssr.path_is_elementary && extp.path_is_elementary
            && dssr.optimal_rc < 1e20 && extp.optimal_rc < 1e20) {
            if (std::abs(dssr.optimal_rc - extp.optimal_rc) > 0.01) {
                verify_status = "L3_CROSS_MISMATCH";
                errors++;
            }
        }

        std::cout << std::left  << std::setw(22) << inst.name
                  << std::right << std::setw(3)  << inst.n_customers
                  << std::setw(11) << buf_orc
                  << std::setw(11) << buf_dssr
                  << std::setw(9)  << std::fixed << std::setprecision(1) << dssr.time_ms
                  << std::setw(9)  << dssr.labels_created
                  << std::setw(5)  << dssr.dssr_iterations
                  << std::setw(5)  << dssr.critical_count
                  << std::setw(5)  << (dssr.path_is_elementary ? "Y" : "N")
                  << std::setw(11) << buf_extp
                  << std::setw(9)  << std::fixed << std::setprecision(1) << extp.time_ms
                  << std::setw(9)  << extp.labels_created
                  << std::setw(5)  << (extp.path_is_elementary ? "Y" : "N")
                  << std::setw(11) << buf_pool
                  << std::setw(9)  << std::fixed << std::setprecision(1) << pool.time_ms
                  << "  " << std::left << std::setw(9) << buf_speedup
                  << verify_status << std::endl;
    }
    std::cout << std::string(150, '-') << "\n";

    return errors;
}

// ── Random stress test ───────────────────────────────────────────────────────
// Generate random Euclidean instances, compare all solvers.
int run_stress_test(int count, int max_n) {
    std::cout << "ESPPRC Stress Test: " << count << " random instances, n <= " << max_n << "\n";
    std::cout << std::string(80, '=') << "\n";

    std::mt19937 rng(42);
    int pass = 0, fail = 0;

    for (int trial = 0; trial < count; trial++) {
        int n_cust = 3 + rng() % (max_n - 2);
        int N = n_cust + 1;

        std::uniform_int_distribution<int> coord_dist(0, 100);
        std::vector<int> cx(N), cy(N);
        for (int i = 0; i < N; i++) {
            cx[i] = coord_dist(rng);
            cy[i] = coord_dist(rng);
        }
        std::vector<std::vector<int>> dist(N, std::vector<int>(N));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int dx = cx[i] - cx[j], dy = cy[i] - cy[j];
                dist[i][j] = (int)std::round(std::sqrt(dx*dx + dy*dy));
            }

        std::uniform_int_distribution<int> dem_dist(1, 5);
        std::vector<int> demand(N);
        demand[0] = 0;
        int total_dem = 0;
        for (int i = 1; i < N; i++) {
            demand[i] = dem_dist(rng);
            total_dem += demand[i];
        }
        int capacity = std::max(3, total_dem * (30 + (int)(rng() % 40)) / 100);

        std::uniform_real_distribution<double> dual_dist(0, 50);
        std::vector<double> cover_duals(n_cust);
        for (int i = 0; i < n_cust; i++)
            cover_duals[i] = dual_dist(rng);

        double tl = 10000.0;
        espprc::Result uni = espprc::solve(dist, demand, capacity, cover_duals, tl);

        // Skip if exact timed out
        if (uni.timed_out) continue;

        bool ok = true;
        std::string err;

        // Test ng-route uni and bidir
        for (int delta : {8, n_cust}) {
            espprc_ng::Result ng = espprc_ng::solve(
                dist, demand, capacity, cover_duals, delta, tl);
            espprc_ng_bidir::Result ngb = espprc_ng_bidir::solve(
                dist, demand, capacity, cover_duals, delta, tl);
            espprc_ng_bidir_par::Result ngbp = espprc_ng_bidir_par::solve(
                dist, demand, capacity, cover_duals, delta, tl);

            if (ng.timed_out || ngb.timed_out) continue;

            // ng_uni is a relaxation: ng_rc <= exact_rc
            if (uni.optimal_rc < 1e20 && ng.optimal_rc < 1e20) {
                if (ng.optimal_rc > uni.optimal_rc + 0.01) {
                    ok = false;
                    err = "ng_uni(d=" + std::to_string(delta) + ") RC="
                        + std::to_string(ng.optimal_rc) + " > exact RC="
                        + std::to_string(uni.optimal_rc);
                    break;
                }
            }

            // ng_bidir is also a relaxation: ngb_rc <= exact_rc
            if (uni.optimal_rc < 1e20 && ngb.optimal_rc < 1e20) {
                if (ngb.optimal_rc > uni.optimal_rc + 0.01) {
                    ok = false;
                    err = "ng_bidir(d=" + std::to_string(delta) + ") RC="
                        + std::to_string(ngb.optimal_rc) + " > exact RC="
                        + std::to_string(uni.optimal_rc);
                    break;
                }
            }

            // When delta=n, ng_uni should equal exact
            if (delta == n_cust) {
                if (uni.optimal_rc < 1e20 && ng.optimal_rc < 1e20) {
                    if (std::abs(uni.optimal_rc - ng.optimal_rc) > 0.01) {
                        ok = false;
                        err = "ng_uni(d=n) RC=" + std::to_string(ng.optimal_rc)
                            + " != exact RC=" + std::to_string(uni.optimal_rc);
                        break;
                    }
                }
            }

            // Verify paths
            if (ng.optimal_rc < 1e20) {
                std::string path_err = verify_path(ng.optimal_path, ng.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "ng_uni(d=" + std::to_string(delta) + ") path: " + path_err;
                    break;
                }
            }
            if (ngb.optimal_rc < 1e20) {
                std::string path_err = verify_path(ngb.optimal_path, ngb.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "ng_bidir(d=" + std::to_string(delta) + ") path: " + path_err;
                    break;
                }
            }

            // ng_bidir_par: must match sequential ng_bidir
            if (!ngbp.timed_out && ngb.optimal_rc < 1e20 && ngbp.optimal_rc < 1e20) {
                if (std::abs(ngb.optimal_rc - ngbp.optimal_rc) > 0.01) {
                    ok = false;
                    err = "ng_bidir_par(d=" + std::to_string(delta) + ") RC="
                        + std::to_string(ngbp.optimal_rc) + " != seq RC="
                        + std::to_string(ngb.optimal_rc);
                    break;
                }
            }
            if (!ngbp.timed_out && ngbp.optimal_rc < 1e20) {
                std::string path_err = verify_path(ngbp.optimal_path, ngbp.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "ng_bidir_par(d=" + std::to_string(delta) + ") path: " + path_err;
                    break;
                }
            }

            // ng+CB: completion bounds is pure pruning.
            // When ng finds a negative RC path, ng_cb must find it too.
            // When both RCs are >= 0, no negative RC column exists — the exact
            // positive value doesn't matter, so mismatch is benign.
            espprc_ng_cb::Result cb = espprc_ng_cb::solve(
                dist, demand, capacity, cover_duals, delta, tl);
            if (!cb.timed_out && ng.optimal_rc < 1e20 && cb.optimal_rc < 1e20) {
                bool ng_negative = ng.optimal_rc < -1e-6;
                if (ng_negative && std::abs(ng.optimal_rc - cb.optimal_rc) > 0.01) {
                    ok = false;
                    err = "ng_cb(d=" + std::to_string(delta) + ") RC="
                        + std::to_string(cb.optimal_rc) + " != ng RC="
                        + std::to_string(ng.optimal_rc)
                        + " (ng found negative RC but cb disagrees)";
                    break;
                }
            }
            if (!cb.timed_out && cb.optimal_rc < 1e20) {
                std::string path_err = verify_path(cb.optimal_path, cb.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "ng_cb(d=" + std::to_string(delta) + ") path: " + path_err;
                    break;
                }
            }

            // ng+Bucket: same algorithm, different data structure. RC must match ng.
            espprc_ng_bucket::Result bkt = espprc_ng_bucket::solve(
                dist, demand, capacity, cover_duals, delta, 20, tl);
            if (!bkt.timed_out && ng.optimal_rc < 1e20 && bkt.optimal_rc < 1e20) {
                if (std::abs(ng.optimal_rc - bkt.optimal_rc) > 0.01) {
                    ok = false;
                    err = "ng_bucket(d=" + std::to_string(delta) + ") RC="
                        + std::to_string(bkt.optimal_rc) + " != ng RC="
                        + std::to_string(ng.optimal_rc);
                    break;
                }
            }
            if (!bkt.timed_out && bkt.optimal_rc < 1e20) {
                std::string path_err = verify_path(bkt.optimal_path, bkt.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "ng_bucket(d=" + std::to_string(delta) + ") path: " + path_err;
                    break;
                }
            }

            // ng+RC-sort: same algorithm, RC-sorted dominance.
            // RC must match ng exactly.
            espprc_ng_rcsort::Result rcs = espprc_ng_rcsort::solve(
                dist, demand, capacity, cover_duals, delta, tl);
            if (!rcs.timed_out && ng.optimal_rc < 1e20 && rcs.optimal_rc < 1e20) {
                if (std::abs(ng.optimal_rc - rcs.optimal_rc) > 0.01) {
                    ok = false;
                    err = "ng_rcsort(d=" + std::to_string(delta) + ") RC="
                        + std::to_string(rcs.optimal_rc) + " != ng RC="
                        + std::to_string(ng.optimal_rc);
                    break;
                }
            }
            if (!rcs.timed_out && rcs.optimal_rc < 1e20) {
                std::string path_err = verify_path(rcs.optimal_path, rcs.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "ng_rcsort(d=" + std::to_string(delta) + ") path: " + path_err;
                    break;
                }
            }

            // ng+Bucket v2: same algorithm, bucket+RC-sorted dominance.
            // RC must match ng exactly.
            espprc_ng_bucket2::Result bk2 = espprc_ng_bucket2::solve(
                dist, demand, capacity, cover_duals, delta, 20, tl);
            if (!bk2.timed_out && ng.optimal_rc < 1e20 && bk2.optimal_rc < 1e20) {
                if (std::abs(ng.optimal_rc - bk2.optimal_rc) > 0.01) {
                    ok = false;
                    err = "ng_bucket2(d=" + std::to_string(delta) + ") RC="
                        + std::to_string(bk2.optimal_rc) + " != ng RC="
                        + std::to_string(ng.optimal_rc);
                    break;
                }
            }
            if (!bk2.timed_out && bk2.optimal_rc < 1e20) {
                std::string path_err = verify_path(bk2.optimal_path, bk2.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "ng_bucket2(d=" + std::to_string(delta) + ") path: " + path_err;
                    break;
                }
            }

            // ng+Pool: same as bucket2 with pre-allocated memory.
            // RC must match ng exactly.
            espprc_ng_pool::Result pool = espprc_ng_pool::solve(
                dist, demand, capacity, cover_duals, delta, 20, tl);
            if (!pool.timed_out && ng.optimal_rc < 1e20 && pool.optimal_rc < 1e20) {
                if (std::abs(ng.optimal_rc - pool.optimal_rc) > 0.01) {
                    ok = false;
                    err = "ng_pool(d=" + std::to_string(delta) + ") RC="
                        + std::to_string(pool.optimal_rc) + " != ng RC="
                        + std::to_string(ng.optimal_rc);
                    break;
                }
            }
            if (!pool.timed_out && pool.optimal_rc < 1e20) {
                std::string path_err = verify_path(pool.optimal_path, pool.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "ng_pool(d=" + std::to_string(delta) + ") path: " + path_err;
                    break;
                }
            }

            // ng+DSSR: tighter than ng (critical nodes never forgotten).
            // dssr_rc >= ng_rc (DSSR is less relaxed).
            // When delta=n, DSSR should match exact.
            espprc_ng_dssr::Result dssr = espprc_ng_dssr::solve(
                dist, demand, capacity, cover_duals, delta, 20, tl);
            if (!dssr.timed_out && ng.optimal_rc < 1e20 && dssr.optimal_rc < 1e20) {
                // DSSR is tighter: dssr_rc >= ng_rc
                if (dssr.optimal_rc < ng.optimal_rc - 0.01) {
                    ok = false;
                    err = "ng_dssr(d=" + std::to_string(delta) + ") RC="
                        + std::to_string(dssr.optimal_rc) + " < ng RC="
                        + std::to_string(ng.optimal_rc)
                        + " (DSSR should be tighter)";
                    break;
                }
                // When delta=n, DSSR should match exact
                if (delta == n_cust) {
                    if (std::abs(dssr.optimal_rc - uni.optimal_rc) > 0.01) {
                        ok = false;
                        err = "ng_dssr(d=n) RC=" + std::to_string(dssr.optimal_rc)
                            + " != exact RC=" + std::to_string(uni.optimal_rc);
                        break;
                    }
                }
            }
            if (!dssr.timed_out && dssr.optimal_rc < 1e20) {
                std::string path_err = verify_path(dssr.optimal_path, dssr.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "ng_dssr(d=" + std::to_string(delta) + ") path: " + path_err;
                    break;
                }
            }

            // DSSR + Pool: must match old DSSR result
            espprc_dssr_pool::Result dssr_p = espprc_dssr_pool::solve(
                dist, demand, capacity, cover_duals, delta, 20, 20, tl);
            if (!dssr_p.timed_out && !dssr.timed_out
                && dssr_p.optimal_rc < 1e20 && dssr.optimal_rc < 1e20) {
                if (std::abs(dssr_p.optimal_rc - dssr.optimal_rc) > 0.01) {
                    ok = false;
                    err = "dssr_pool RC=" + std::to_string(dssr_p.optimal_rc)
                        + " != dssr RC=" + std::to_string(dssr.optimal_rc);
                    break;
                }
            }
            if (!dssr_p.timed_out && dssr_p.optimal_rc < 1e20) {
                std::string path_err = verify_path(dssr_p.optimal_path, dssr_p.optimal_rc,
                    dist, demand, capacity, cover_duals, false);
                if (!path_err.empty()) {
                    ok = false;
                    err = "dssr_pool path: " + path_err;
                    break;
                }
            }

            // Exact + Pool: must match exact oracle
            espprc_exact_pool::Result extp = espprc_exact_pool::solve(
                dist, demand, capacity, cover_duals, 20, tl);
            if (!extp.timed_out && extp.optimal_rc < 1e20) {
                if (std::abs(extp.optimal_rc - uni.optimal_rc) > 0.01) {
                    ok = false;
                    err = "exact_pool RC=" + std::to_string(extp.optimal_rc)
                        + " != exact RC=" + std::to_string(uni.optimal_rc);
                    break;
                }
                std::string path_err = verify_path(extp.optimal_path, extp.optimal_rc,
                    dist, demand, capacity, cover_duals, true);
                if (!path_err.empty()) {
                    ok = false;
                    err = "exact_pool path: " + path_err;
                    break;
                }
            }
        }

        if (ok) {
            pass++;
        } else {
            fail++;
            std::cout << "FAIL #" << trial << " (n=" << n_cust << ", Q=" << capacity << "): "
                      << err << std::endl;
            if (fail <= 5) {
                std::cout << "  duals=[";
                for (int i = 0; i < n_cust; i++)
                    std::cout << (i ? "," : "") << std::fixed << std::setprecision(2) << cover_duals[i];
                std::cout << "]\n";
            }
        }
    }

    std::cout << std::string(80, '-') << "\n";
    std::cout << "Stress test: " << pass << " passed, " << fail << " failed\n";

    return fail;
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string inst_dir;
    int max_n = 9999;
    int stress_count = 0;
    int stress_max_n = 12;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--up-to" && i + 1 < argc)
            max_n = std::stoi(argv[++i]);
        else if (arg == "--stress" && i + 1 < argc)
            stress_count = std::stoi(argv[++i]);
        else if (arg == "--max-n" && i + 1 < argc)
            stress_max_n = std::stoi(argv[++i]);
        else
            inst_dir = arg;
    }

    int errors = 0;

    // Stress test mode
    if (stress_count > 0) {
        errors = run_stress_test(stress_count, stress_max_n);
        return errors > 0 ? 1 : 0;
    }

    // JSON benchmark mode
    if (inst_dir.empty()) {
        for (auto& c : {
            "studies/espprc_testset/instances",
            "../../../studies/espprc_testset/instances",
            "../../../../studies/espprc_testset/instances",
        }) {
            if (fs::exists(c)) { inst_dir = c; break; }
        }
    }
    if (inst_dir.empty() || !fs::exists(inst_dir)) {
        std::cerr << "Instance directory not found.\n";
        return 1;
    }

    errors = run_json_benchmark(inst_dir, max_n);

    if (errors > 0) {
        std::cout << "\n** " << errors << " VERIFICATION ERRORS **\n";
        return 1;
    }

    return 0;
}
