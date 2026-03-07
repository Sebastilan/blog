// ============================================================================
// bench_exact_final.cpp — Final comparison: 5 solvers × all testset instances
// ============================================================================
// Solvers compared:
//   A. exact_pool + CB        (Pool+Bucket+CB, unidirectional)
//   B. exact_pool (no CB)     (Pool+Bucket, unidirectional)
//   C. dssr_pool + CB         (DSSR iterative, with CB)
//   D. exact_modular fullcfg  (Pool+Bucket+Bidir+Parallel+CB = C8+CB)
//   E. exact_modular no CB    (Pool+Bucket+Bidir+Parallel, no CB = C8)
//   F. bidir_pool (relaxed)   (ng-bidir + CB, for reference)
//
// Usage:
//   bench_exact_final [instances_dir]
// ============================================================================

#include "exact/espprc_exact_pool.h"
#include "exact/espprc_dssr_pool.h"
#include "exact/espprc_exact_modular.h"
#include "relaxed/espprc_ng_bidir_pool.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <iomanip>
#include <cmath>
#include <map>

namespace fs = std::filesystem;
using json = nlohmann::json;

static const double TIME_LIMIT_MS = 60000.0;

// Ordered instance list
static const char* INSTANCE_NAMES[] = {
    "tiny3_initial",
    "tiny5_random",
    "small8_random",
    "en13k4_iter1",
    "en13k4_iter2",
    "en13k4_converged",
    "pn16k8_iter1",
    "pn16k8_converged",
    "pn19k2_initial",
    "pn19k2_perturbed",
    "pn22k2_initial",
    "pn22k2_perturbed",
    "an32k5_initial",
    "an32k5_perturbed",
};
static const int N_INST = 14;

struct Instance {
    std::string name;
    int n_cust;
    int capacity;
    std::vector<std::vector<int>> dist;
    std::vector<int> demand;
    std::vector<double> duals;
    double oracle_rc = 1e30;
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
    return true;
}

struct Row {
    std::string inst;
    int n;
    // [solver][field]: time_ms, labels, pruned, rc, timed_out
    struct SR {
        double time_ms = -1;
        int labels = 0;
        long long pruned = 0;
        double rc = 1e30;
        bool timed_out = false;
        bool skipped = false;
    };
    SR A, B, C, D, E, F;
    double oracle;
};

static std::string fmt_ms(double ms, bool timed_out, bool skipped) {
    if (skipped)   return "SKIP";
    if (timed_out) return "TIMEOUT";
    char buf[32];
    if (ms < 1.0)       snprintf(buf, sizeof(buf), "%.3fms", ms);
    else if (ms < 1000) snprintf(buf, sizeof(buf), "%.1fms", ms);
    else                snprintf(buf, sizeof(buf), "%.2fs", ms/1000.0);
    return buf;
}

static std::string fmt_rc(double rc, double oracle, bool timed_out, bool skipped) {
    if (skipped) return "-";
    char buf[40];
    if (rc > 1e29) { snprintf(buf, sizeof(buf), "none"); return buf; }
    if (oracle < 1e29 && !timed_out && std::abs(rc - oracle) > 1e-4)
        snprintf(buf, sizeof(buf), "%.2f(!)", rc);
    else if (timed_out)
        snprintf(buf, sizeof(buf), "%.2f*", rc);
    else
        snprintf(buf, sizeof(buf), "%.2f", rc);
    return buf;
}

int main(int argc, char* argv[]) {
    std::string inst_dir = "testset/instances";
    if (argc >= 2) inst_dir = argv[1];

    // Load oracles from manifest
    std::map<std::string, double> oracle_map;
    {
        fs::path mp = fs::path(inst_dir) / "manifest.json";
        std::ifstream mf(mp);
        if (mf) {
            json mj = json::parse(mf);
            if (mj.is_array())
                for (auto& e : mj) {
                    std::string n = e.value("name", "");
                    double r = e.value("rc", 1e30);
                    if (!n.empty()) oracle_map[n] = r;
                }
        }
    }

    // Load instances
    std::vector<Instance> insts;
    for (int i = 0; i < N_INST; i++) {
        Instance inst;
        inst.name = INSTANCE_NAMES[i];
        fs::path p = fs::path(inst_dir) / (inst.name + ".json");
        if (!load_instance(p, inst)) continue;
        if (oracle_map.count(inst.name)) inst.oracle_rc = oracle_map[inst.name];
        insts.push_back(inst);
    }

    std::vector<Row> rows;

    for (auto& inst : insts) {
        std::cerr << "\n=== " << inst.name << " (n=" << inst.n_cust << ") ===\n";
        Row row;
        row.inst   = inst.name;
        row.n      = inst.n_cust;
        row.oracle = inst.oracle_rc;

        auto& d  = inst.dist;
        auto& dm = inst.demand;
        int   Q  = inst.capacity;
        auto& du = inst.duals;

        // A. exact_pool + CB
        {
            std::cerr << "  A. exact_pool+CB... " << std::flush;
            auto r = espprc_exact_pool::solve(d, dm, Q, du, 20, TIME_LIMIT_MS, true);
            row.A.time_ms = r.time_ms; row.A.labels = r.labels_created;
            row.A.pruned  = r.labels_pruned; row.A.rc = r.optimal_rc;
            row.A.timed_out = r.timed_out;
            std::cerr << fmt_ms(r.time_ms, r.timed_out, false) << "\n";
        }

        // B. exact_pool no CB
        {
            std::cerr << "  B. exact_pool(noCB)... " << std::flush;
            auto r = espprc_exact_pool::solve(d, dm, Q, du, 20, TIME_LIMIT_MS, false);
            row.B.time_ms = r.time_ms; row.B.labels = r.labels_created;
            row.B.pruned  = 0; row.B.rc = r.optimal_rc;
            row.B.timed_out = r.timed_out;
            std::cerr << fmt_ms(r.time_ms, r.timed_out, false) << "\n";
        }

        // C. dssr_pool + CB (skip n>25 if already can't)
        if (inst.n_cust <= 25) {
            std::cerr << "  C. dssr_pool+CB... " << std::flush;
            auto r = espprc_dssr_pool::solve(d, dm, Q, du, 8, 20, 20, TIME_LIMIT_MS, true);
            row.C.time_ms = r.time_ms; row.C.labels = r.labels_created;
            row.C.pruned  = r.labels_pruned; row.C.rc = r.optimal_rc;
            row.C.timed_out = r.timed_out;
            std::cerr << fmt_ms(r.time_ms, r.timed_out, false) << "\n";
        } else {
            row.C.skipped = true;
            std::cerr << "  C. dssr_pool+CB... SKIP (n>25)\n";
        }

        // D. exact_modular full (Pool+Bucket+Bidir+Parallel+CB)
        {
            std::cerr << "  D. modular full+CB... " << std::flush;
            auto r = espprc_exact_modular::solve(d, dm, Q, du,
                true, true, true, true, true, 20, TIME_LIMIT_MS);
            row.D.time_ms = r.time_ms; row.D.labels = r.labels_created;
            row.D.pruned  = r.labels_pruned; row.D.rc = r.optimal_rc;
            row.D.timed_out = r.timed_out;
            std::cerr << fmt_ms(r.time_ms, r.timed_out, false) << "\n";
        }

        // E. exact_modular full no CB
        {
            std::cerr << "  E. modular full(noCB)... " << std::flush;
            auto r = espprc_exact_modular::solve(d, dm, Q, du,
                true, true, true, true, false, 20, TIME_LIMIT_MS);
            row.E.time_ms = r.time_ms; row.E.labels = r.labels_created;
            row.E.pruned  = 0; row.E.rc = r.optimal_rc;
            row.E.timed_out = r.timed_out;
            std::cerr << fmt_ms(r.time_ms, r.timed_out, false) << "\n";
        }

        // F. bidir_pool relaxed (ng-bidir + CB, reference)
        {
            std::cerr << "  F. bidir_pool(relax)... " << std::flush;
            auto r = espprc_ng_bidir_pool::solve(d, dm, Q, du, 8, 20, TIME_LIMIT_MS);
            row.F.time_ms = r.time_ms;
            row.F.labels  = r.fwd_labels + r.bwd_labels;
            row.F.pruned  = r.labels_pruned;
            row.F.rc      = r.optimal_rc;
            row.F.timed_out = r.timed_out;
            std::cerr << fmt_ms(r.time_ms, r.timed_out, false) << "\n";
        }

        rows.push_back(row);
    }

    // ── Print table ───────────────────────────────────────────────────────────
    std::cout << "\n";
    std::cout << "=============================================================================\n";
    std::cout << "  ESPPRC Final Comparison (timeout=60s)\n";
    std::cout << "  A=exact+CB  B=exact(noCB)  C=dssr+CB  D=mod_full+CB  E=mod_full(noCB)  F=bidir(relax)\n";
    std::cout << "=============================================================================\n\n";

    // Per-instance table
    for (auto& row : rows) {
        std::cout << "--- " << row.inst << " (n=" << row.n
                  << ", oracle=" << (row.oracle < 1e29 ? std::to_string((int)std::round(row.oracle)) : "?")
                  << ") ---\n";

        auto pr = [&](const char* label, const Row::SR& s) {
            std::cout << std::left << std::setw(22) << label
                      << "  time=" << std::setw(10) << fmt_ms(s.time_ms, s.timed_out, s.skipped)
                      << "  labels=" << std::setw(8);
            if (s.skipped) std::cout << "-";
            else {
                char buf[20]; snprintf(buf, sizeof(buf), "%d", s.labels);
                std::cout << buf;
            }
            std::cout << "  pruned=" << std::setw(8);
            if (s.skipped || s.pruned == 0) std::cout << (s.skipped ? "-" : "0");
            else {
                char buf[20]; snprintf(buf, sizeof(buf), "%lld", s.pruned);
                std::cout << buf;
            }
            std::cout << "  rc=" << fmt_rc(s.rc, row.oracle, s.timed_out, s.skipped) << "\n";
        };

        pr("A exact_pool+CB",   row.A);
        pr("B exact_pool",      row.B);
        pr("C dssr+CB",         row.C);
        pr("D mod_full+CB",     row.D);
        pr("E mod_full",        row.E);
        pr("F bidir(relax)",    row.F);
        std::cout << "\n";
    }

    // ── CSV ───────────────────────────────────────────────────────────────────
    std::cout << "\n--- CSV ---\n";
    std::cout << "inst,n,oracle,solver,time_ms,labels,pruned,rc,timed_out\n";

    auto emit_csv = [&](const Row& row, const char* sname, const Row::SR& s) {
        if (s.skipped) {
            std::cout << row.inst << "," << row.n << ","
                      << (row.oracle < 1e29 ? row.oracle : -999) << ","
                      << sname << ",SKIP,SKIP,SKIP,SKIP,SKIP\n";
            return;
        }
        std::cout << row.inst << "," << row.n << ","
                  << (row.oracle < 1e29 ? row.oracle : -999) << ","
                  << sname << ","
                  << std::fixed << std::setprecision(2) << s.time_ms << ","
                  << s.labels << "," << s.pruned << ","
                  << std::setprecision(4) << s.rc << ","
                  << (s.timed_out ? "true" : "false") << "\n";
    };

    for (auto& row : rows) {
        emit_csv(row, "A_exact_cb",    row.A);
        emit_csv(row, "B_exact_nocb",  row.B);
        emit_csv(row, "C_dssr_cb",     row.C);
        emit_csv(row, "D_mod_full_cb", row.D);
        emit_csv(row, "E_mod_full",    row.E);
        emit_csv(row, "F_bidir_relax", row.F);
    }

    return 0;
}
