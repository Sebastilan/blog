// ============================================================================
// bench_exact_ablation.cpp — ESPPRC Exact Strategy Ablation Benchmark
// ============================================================================
// Runs 9 configurations (C0-C8) x 4 test instances with progressive cutoff.
// Verifies correctness against manifest oracle RC values.
//
// Usage:
//   bench_exact_ablation [instances_dir]
//
// Configurations:
//   C0: pool=0 bucket=0 bidir=0 parallel=0  (bare baseline)
//   C1: pool=1 bucket=0 bidir=0 parallel=0
//   C2: pool=0 bucket=1 bidir=0 parallel=0
//   C3: pool=0 bucket=0 bidir=1 parallel=0
//   C4: pool=1 bucket=1 bidir=0 parallel=0  (== EXACT_POOL)
//   C5: pool=1 bucket=0 bidir=1 parallel=0
//   C6: pool=0 bucket=1 bidir=1 parallel=0
//   C7: pool=1 bucket=1 bidir=1 parallel=0
//   C8: pool=1 bucket=1 bidir=1 parallel=1
// ============================================================================

#include "exact/espprc_exact_modular.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <iomanip>
#include <cmath>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Instance ─────────────────────────────────────────────────────────────────
struct Instance {
    std::string name;
    std::string file;
    int n;
    std::vector<std::vector<int>> dist;
    std::vector<int> demand;
    int capacity;
    std::vector<double> cover_duals;
    double oracle_rc;
};

static Instance load_instance(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    json j;
    f >> j;

    Instance inst;
    inst.name     = j.value("name", fs::path(path).stem().string());
    inst.n        = j["n_customers"].get<int>();
    inst.capacity = j["capacity"].get<int>();

    for (auto& row : j["dist"])
        inst.dist.push_back(row.get<std::vector<int>>());
    inst.demand      = j["demand"].get<std::vector<int>>();
    inst.cover_duals = j["cover_duals"].get<std::vector<double>>();

    inst.oracle_rc = 1e30;
    if (j.contains("oracle") && j["oracle"].contains("optimal_rc"))
        inst.oracle_rc = j["oracle"]["optimal_rc"].get<double>();

    return inst;
}

// ── Config definition ─────────────────────────────────────────────────────────
struct Config {
    int  id;
    bool pool, bucket, bidir, parallel;
    std::string label;
};

static const Config CONFIGS[] = {
    {0, false, false, false, false, "C0 (baseline)"},
    {1, true,  false, false, false, "C1 (pool)"},
    {2, false, true,  false, false, "C2 (bucket)"},
    {3, false, false, true,  false, "C3 (bidir)"},
    {4, true,  true,  false, false, "C4 (pool+bucket=EXACT_POOL)"},
    {5, true,  false, true,  false, "C5 (pool+bidir)"},
    {6, false, true,  true,  false, "C6 (bucket+bidir)"},
    {7, true,  true,  true,  false, "C7 (all-nopar)"},
    {8, true,  true,  true,  true,  "C8 (full)"},
};
static constexpr int N_CONFIGS = 9;

// ── Cell result ───────────────────────────────────────────────────────────────
struct Cell {
    bool   skipped  = false;
    bool   timed_out= false;
    bool   rc_ok    = false;
    double time_ms  = 0;
    int    labels   = 0;
    long long dom_checks = 0;
    double optimal_rc = 1e30;
};

// ── CSV output ────────────────────────────────────────────────────────────────
static void write_csv(
    const std::vector<std::string>& inst_names,
    const std::vector<std::vector<Cell>>& results,  // [config][inst]
    const std::string& out_path)
{
    std::ofstream f(out_path);
    // Header
    f << "config,pool,bucket,bidir,parallel";
    for (auto& name : inst_names) {
        f << "," << name << "_time_ms"
          << "," << name << "_labels"
          << "," << name << "_dom_checks"
          << "," << name << "_rc_ok"
          << "," << name << "_status";
    }
    f << "\n";

    for (int ci = 0; ci < N_CONFIGS; ci++) {
        const Config& c = CONFIGS[ci];
        f << "C" << c.id << ","
          << c.pool << "," << c.bucket << "," << c.bidir << "," << c.parallel;
        for (int ii = 0; ii < (int)inst_names.size(); ii++) {
            const Cell& cell = results[ci][ii];
            std::string status = cell.skipped ? "SKIP"
                               : cell.timed_out ? "TIMEOUT"
                               : cell.rc_ok ? "OK" : "RC_ERR";
            f << "," << (cell.skipped || cell.timed_out ? "" : std::to_string((int)std::round(cell.time_ms)))
              << "," << (cell.skipped ? "" : std::to_string(cell.labels))
              << "," << (cell.skipped ? "" : std::to_string(cell.dom_checks))
              << "," << (cell.skipped ? "" : (cell.rc_ok ? "1" : "0"))
              << "," << status;
        }
        f << "\n";
    }
    std::cout << "[CSV] Written to: " << out_path << "\n";
}

// ── Terminal table ────────────────────────────────────────────────────────────
static void print_table(
    const std::vector<Instance>& instances,
    const std::vector<std::vector<Cell>>& results)
{
    int n_inst = (int)instances.size();

    // Header
    std::cout << "\n";
    std::cout << std::left << std::setw(28) << "Config";
    for (auto& inst : instances)
        std::cout << std::right << std::setw(12) << inst.name.substr(0,10)
                  << std::setw(8) << "labels";
    std::cout << "\n";
    std::cout << std::string(28 + n_inst * 20, '-') << "\n";

    for (int ci = 0; ci < N_CONFIGS; ci++) {
        const Config& c = CONFIGS[ci];
        std::cout << std::left << std::setw(28) << c.label;
        for (int ii = 0; ii < n_inst; ii++) {
            const Cell& cell = results[ci][ii];
            if (cell.skipped) {
                std::cout << std::right << std::setw(12) << "SKIP" << std::setw(8) << "-";
            } else if (cell.timed_out) {
                std::cout << std::right << std::setw(12) << "TIMEOUT" << std::setw(8) << "-";
            } else {
                std::string time_str = std::to_string((int)std::round(cell.time_ms)) + "ms"
                                     + (cell.rc_ok ? "" : "!");
                std::cout << std::right << std::setw(12) << time_str
                          << std::setw(8) << cell.labels;
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    std::cout << "Note: '!' = RC mismatch vs oracle\n";
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    std::string inst_dir = "testset/instances";
    if (argc >= 2) inst_dir = argv[1];

    // ── Target instances (in order of n) ──────────────────────────────
    static const char* TARGET_FILES[] = {
        "en13k4_iter1.json",
        "pn16k8_iter1.json",
        "pn19k2_initial.json",
        "pn22k2_initial.json",
    };
    static constexpr int N_INSTS = 4;

    std::vector<Instance> instances;
    for (auto* fname : TARGET_FILES) {
        std::string path = inst_dir + "/" + fname;
        try {
            instances.push_back(load_instance(path));
            std::cout << "[Load] " << instances.back().name
                      << " (n=" << instances.back().n << ")\n";
        } catch (std::exception& e) {
            std::cerr << "[WARN] Skipping " << fname << ": " << e.what() << "\n";
        }
    }

    int n_inst = (int)instances.size();
    if (n_inst == 0) {
        std::cerr << "[ERROR] No instances loaded from: " << inst_dir << "\n";
        return 1;
    }

    std::vector<std::string> inst_names;
    for (auto& inst : instances) inst_names.push_back(inst.name);

    // ── Ablation matrix ────────────────────────────────────────────────
    // results[config_idx][inst_idx]
    std::vector<std::vector<Cell>> results(N_CONFIGS, std::vector<Cell>(n_inst));

    static constexpr double TIME_LIMIT_MS = 60000.0;
    static constexpr int    THETA         = 20;
    static constexpr double RC_TOL        = 1e-4;

    std::cout << "\n=== ESPPRC Exact Ablation (9 configs x " << n_inst << " instances) ===\n\n";

    for (int ci = 0; ci < N_CONFIGS; ci++) {
        const Config& cfg = CONFIGS[ci];
        std::cout << "--- " << cfg.label << " ---\n";

        bool prev_timed_out = false;

        for (int ii = 0; ii < n_inst; ii++) {
            Cell& cell = results[ci][ii];
            const Instance& inst = instances[ii];

            // Progressive cutoff: skip if previous n timed out
            if (prev_timed_out) {
                cell.skipped = true;
                std::cout << "  " << inst.name << ": SKIP (prev timeout)\n";
                continue;
            }

            std::cout << "  " << inst.name << " (n=" << inst.n << ")... " << std::flush;

            using namespace espprc_exact_modular;
            Result r = solve(
                inst.dist, inst.demand, inst.capacity, inst.cover_duals,
                cfg.pool, cfg.bucket, cfg.bidir, cfg.parallel,
                THETA, TIME_LIMIT_MS);

            cell.timed_out  = r.timed_out;
            cell.time_ms    = r.time_ms;
            cell.labels     = r.labels_created;
            cell.dom_checks = r.dom_checks;
            cell.optimal_rc = r.optimal_rc;

            if (r.timed_out) {
                std::cout << "TIMEOUT (" << (int)r.time_ms << "ms)\n";
                prev_timed_out = true;
                continue;
            }

            // Correctness check
            if (inst.oracle_rc < 1e29) {
                double diff = std::abs(r.optimal_rc - inst.oracle_rc);
                cell.rc_ok = (diff < RC_TOL);
                if (!cell.rc_ok) {
                    std::cout << "RC_ERR (got=" << r.optimal_rc
                              << " oracle=" << inst.oracle_rc << ") ";
                }
            } else {
                cell.rc_ok = true; // no oracle, assume ok
            }

            std::cout << (int)std::round(r.time_ms) << "ms"
                      << " labels=" << r.labels_created
                      << " dom=" << r.dom_checks
                      << " rc=" << std::fixed << std::setprecision(2) << r.optimal_rc
                      << (cell.rc_ok ? " OK" : " !")
                      << "\n";
        }
        std::cout << "\n";
    }

    // ── Print summary table ────────────────────────────────────────────
    print_table(instances, results);

    // ── Write CSV ──────────────────────────────────────────────────────
    write_csv(inst_names, results, "ablation_results.csv");

    // ── Detailed results table ─────────────────────────────────────────
    std::cout << "\n=== Detailed Results Table ===\n";
    std::cout << std::left << std::setw(6) << "Config";
    for (auto& inst : instances) {
        std::string h = inst.name.substr(0,8);
        std::cout << std::right
                  << std::setw(9)  << h + "_t"
                  << std::setw(8)  << h + "_lb"
                  << std::setw(9)  << h + "_dom";
    }
    std::cout << "\n" << std::string(6 + n_inst * 26, '-') << "\n";

    for (int ci = 0; ci < N_CONFIGS; ci++) {
        const Config& c = CONFIGS[ci];
        std::cout << std::left << std::setw(6) << ("C" + std::to_string(c.id));
        for (int ii = 0; ii < n_inst; ii++) {
            const Cell& cell = results[ci][ii];
            if (cell.skipped) {
                std::cout << std::right
                          << std::setw(9) << "SKIP"
                          << std::setw(8) << "-"
                          << std::setw(9) << "-";
            } else if (cell.timed_out) {
                std::cout << std::right
                          << std::setw(9) << "TIMEOUT"
                          << std::setw(8) << "-"
                          << std::setw(9) << "-";
            } else {
                std::cout << std::right
                          << std::setw(9) << ((int)std::round(cell.time_ms))
                          << std::setw(8) << cell.labels
                          << std::setw(9) << cell.dom_checks;
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    return 0;
}
