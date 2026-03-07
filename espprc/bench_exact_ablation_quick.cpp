// ============================================================================
// bench_exact_ablation_quick.cpp — C0/C1/C2 ablation for exact ESPPRC
// ============================================================================
// C0: use_pool=false, use_bucket=false  (bare baseline)
// C1: use_pool=true,  use_bucket=false  (pool only)
// C2: use_pool=false, use_bucket=true   (bucket only)
//
// Usage:
//   bench_exact_ablation_quick [instances_dir]
//
// Outputs a table to stdout.
// Timeout: 60s per run. Progressive cutoff: if n times out, skip larger n.
// ============================================================================

#include "exact/espprc_exact_ablation.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <iomanip>
#include <cmath>

namespace fs = std::filesystem;
using json = nlohmann::json;

static const double TIME_LIMIT_MS = 60000.0;

struct Config {
    const char* name;
    bool use_pool;
    bool use_bucket;
};

static const Config CONFIGS[] = {
    {"C0 (bare)",    false, false},
    {"C1 (pool)",    true,  false},
    {"C2 (bucket)",  false, true },
};
static const int N_CONFIGS = 3;

static const char* INSTANCES[] = {
    "en13k4_iter1",
    "pn16k8_iter1",
    "pn19k2_initial",
    "pn22k2_initial",
};
static const int N_INSTANCES = 4;

struct RunResult {
    double time_ms;
    int    labels;
    long long dom_checks;
    double optimal_rc;
    bool   timed_out;
    bool   skipped;
};

// Load instance from JSON file
struct Instance {
    int n_cust;
    int capacity;
    std::vector<std::vector<int>> dist;
    std::vector<int> demand;
    std::vector<double> duals;
    double oracle_rc;
};

static bool load_instance(const fs::path& path, Instance& inst) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    json j = json::parse(ifs);

    inst.n_cust   = j["n_customers"];
    inst.capacity = j["capacity"];
    int N = inst.n_cust + 1;

    inst.dist.assign(N, std::vector<int>(N));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            inst.dist[i][k] = j["dist"][i][k].get<int>();

    inst.demand.resize(N);
    for (int i = 0; i < N; i++)
        inst.demand[i] = j["demand"][i].get<int>();

    inst.duals.resize(inst.n_cust);
    for (int i = 0; i < inst.n_cust; i++)
        inst.duals[i] = j["cover_duals"][i].get<double>();

    inst.oracle_rc = 1e30;
    return true;
}

static std::string fmt_ms(double ms, bool timed_out) {
    if (timed_out) return "TIMEOUT";
    char buf[32];
    if (ms < 1.0)       snprintf(buf, sizeof(buf), "%.3fms", ms);
    else if (ms < 1000) snprintf(buf, sizeof(buf), "%.1fms", ms);
    else                snprintf(buf, sizeof(buf), "%.1fs", ms / 1000.0);
    return buf;
}

static std::string fmt_count(long long v) {
    char buf[32];
    if (v > 1000000000LL) snprintf(buf, sizeof(buf), "%.1fB", v / 1e9);
    else if (v > 1000000) snprintf(buf, sizeof(buf), "%.1fM", v / 1e6);
    else if (v > 1000)    snprintf(buf, sizeof(buf), "%.0fK", v / 1e3);
    else                  snprintf(buf, sizeof(buf), "%lld", v);
    return buf;
}

int main(int argc, char* argv[]) {
    std::string instances_dir = "testset/instances";
    if (argc >= 2) instances_dir = argv[1];

    // Load manifest for oracle RCs
    std::map<std::string, double> oracle_map;
    {
        fs::path mpath = fs::path(instances_dir) / "manifest.json";
        std::ifstream mf(mpath);
        if (mf) {
            json mj = json::parse(mf);
            if (mj.is_array()) {
                for (auto& entry : mj) {
                    std::string name = entry.value("name", "");
                    double rc = entry.value("rc", 1e30);
                    if (!name.empty()) oracle_map[name] = rc;
                }
            }
        }
    }

    // Load instances
    std::vector<Instance> instances(N_INSTANCES);
    std::vector<bool> inst_ok(N_INSTANCES, false);
    for (int i = 0; i < N_INSTANCES; i++) {
        fs::path p = fs::path(instances_dir) / (std::string(INSTANCES[i]) + ".json");
        inst_ok[i] = load_instance(p, instances[i]);
        if (inst_ok[i] && oracle_map.count(INSTANCES[i]))
            instances[i].oracle_rc = oracle_map[INSTANCES[i]];
    }

    // Run matrix: [config][instance]
    RunResult results[N_CONFIGS][N_INSTANCES];
    for (int c = 0; c < N_CONFIGS; c++) {
        bool skip_rest = false;
        for (int i = 0; i < N_INSTANCES; i++) {
            auto& r = results[c][i];
            r = {0, 0, 0, 1e30, false, false};

            if (skip_rest || !inst_ok[i]) {
                r.skipped = true;
                continue;
            }

            std::cerr << "Running " << CONFIGS[c].name
                      << " on " << INSTANCES[i] << "...\n" << std::flush;

            auto res = espprc_exact_ablation::solve(
                instances[i].dist,
                instances[i].demand,
                instances[i].capacity,
                instances[i].duals,
                CONFIGS[c].use_pool,
                CONFIGS[c].use_bucket,
                20,
                TIME_LIMIT_MS
            );

            r.time_ms    = res.time_ms;
            r.labels     = res.labels_created;
            r.dom_checks = res.dom_checks;
            r.optimal_rc = res.optimal_rc;
            r.timed_out  = res.timed_out;

            // Correctness check
            if (!r.timed_out && instances[i].oracle_rc < 1e29) {
                double diff = std::abs(r.optimal_rc - instances[i].oracle_rc);
                if (diff > 1e-4) {
                    std::cerr << "  *** RC MISMATCH: got " << r.optimal_rc
                              << " oracle=" << instances[i].oracle_rc << "\n";
                }
            }

            if (r.timed_out) skip_rest = true;
        }
    }

    // Print table
    std::cout << "\n";
    std::cout << "=== ESPPRC Exact Ablation: C0/C1/C2 ===\n\n";

    // Header
    std::cout << std::left << std::setw(14) << "Config";
    for (int i = 0; i < N_INSTANCES; i++) {
        std::string hdr = std::string(INSTANCES[i]) + "(n=" +
                          std::to_string(instances[i].n_cust) + ")";
        std::cout << "  " << std::setw(30) << hdr;
    }
    std::cout << "\n";

    std::cout << std::string(14 + N_INSTANCES * 32, '-') << "\n";

    for (int c = 0; c < N_CONFIGS; c++) {
        // time row
        std::cout << std::left << std::setw(14) << CONFIGS[c].name;
        for (int i = 0; i < N_INSTANCES; i++) {
            const auto& r = results[c][i];
            std::string cell;
            if (r.skipped)
                cell = "SKIP";
            else
                cell = fmt_ms(r.time_ms, r.timed_out);
            std::cout << "  " << std::setw(30) << cell;
        }
        std::cout << "\n";

        // labels row
        std::cout << std::left << std::setw(14) << "  labels";
        for (int i = 0; i < N_INSTANCES; i++) {
            const auto& r = results[c][i];
            std::string cell = r.skipped ? "-" : fmt_count(r.labels);
            std::cout << "  " << std::setw(30) << cell;
        }
        std::cout << "\n";

        // dom_checks row
        std::cout << std::left << std::setw(14) << "  dom_chk";
        for (int i = 0; i < N_INSTANCES; i++) {
            const auto& r = results[c][i];
            std::string cell = r.skipped ? "-" : fmt_count(r.dom_checks);
            std::cout << "  " << std::setw(30) << cell;
        }
        std::cout << "\n";

        // rc row
        std::cout << std::left << std::setw(14) << "  rc";
        for (int i = 0; i < N_INSTANCES; i++) {
            const auto& r = results[c][i];
            std::string cell;
            if (r.skipped) cell = "-";
            else {
                char buf[32];
                if (r.timed_out)
                    snprintf(buf, sizeof(buf), "%.1f(partial)", r.optimal_rc);
                else {
                    double oracle = instances[i].oracle_rc;
                    bool ok = oracle > 1e29 || std::abs(r.optimal_rc - oracle) < 1e-4;
                    snprintf(buf, sizeof(buf), "%.1f%s", r.optimal_rc, ok ? " OK" : " MISMATCH");
                }
                cell = buf;
            }
            std::cout << "  " << std::setw(30) << cell;
        }
        std::cout << "\n\n";
    }

    // CSV output for easy copy
    std::cout << "\n--- CSV ---\n";
    std::cout << "config,instance,n,time_ms,labels,dom_checks,optimal_rc,timed_out\n";
    for (int c = 0; c < N_CONFIGS; c++) {
        for (int i = 0; i < N_INSTANCES; i++) {
            const auto& r = results[c][i];
            std::cout << CONFIGS[c].name << ","
                      << INSTANCES[i] << ","
                      << instances[i].n_cust << ",";
            if (r.skipped) {
                std::cout << "SKIP,SKIP,SKIP,SKIP,SKIP\n";
            } else {
                std::cout << std::fixed << std::setprecision(2) << r.time_ms << ","
                          << r.labels << ","
                          << r.dom_checks << ","
                          << std::setprecision(4) << r.optimal_rc << ","
                          << (r.timed_out ? "true" : "false") << "\n";
            }
        }
    }

    return 0;
}
