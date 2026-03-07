// ============================================================================
// ESPPRC Test Runner
// ============================================================================
// Reads JSON test instances, runs the label-setting solver, compares with
// oracle optimal. No LP solver dependency — pure algorithmic test.
//
// Usage: test_espprc [instances_dir] [--up-to N]
//   instances_dir: path to JSON instance files (default: studies/espprc_testset/instances)
//   --up-to N:     only test instances with n_customers <= N
// ============================================================================

#include "exact/espprc.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>
#include <iomanip>

namespace fs = std::filesystem;
using json = nlohmann::json;

static const double TOL = 0.5;  // tolerance for RC comparison (MIP oracle may be inexact)

struct TestInstance {
    std::string name;
    int n_customers;
    std::vector<std::vector<int>> dist;
    std::vector<int> demand;
    int capacity;
    std::vector<double> cover_duals;

    // Oracle
    double oracle_rc;
    std::vector<int> oracle_path;
    int    oracle_cost;
    std::string oracle_method;
};

TestInstance load_instance(const fs::path& filepath) {
    std::ifstream ifs(filepath);
    json j = json::parse(ifs);

    TestInstance inst;
    inst.name        = j["name"];
    inst.n_customers = j["n_customers"];
    inst.capacity    = j["capacity"];

    // dist: 2D array
    auto& jd = j["dist"];
    int N = (int)jd.size();
    inst.dist.resize(N, std::vector<int>(N));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            inst.dist[i][k] = jd[i][k].get<int>();

    // demand
    for (auto& v : j["demand"])
        inst.demand.push_back(v.get<int>());

    // cover_duals
    for (auto& v : j["cover_duals"])
        inst.cover_duals.push_back(v.get<double>());

    // oracle
    auto& oracle = j["oracle"];
    inst.oracle_rc = oracle["optimal_rc"].is_null() ? 1e30 : oracle["optimal_rc"].get<double>();
    inst.oracle_cost = oracle["optimal_cost"].is_null() ? 0 : oracle["optimal_cost"].get<int>();
    inst.oracle_method = oracle["method"].get<std::string>();

    if (!oracle["optimal_path"].is_null())
        for (auto& v : oracle["optimal_path"])
            inst.oracle_path.push_back(v.get<int>());

    return inst;
}

// Verify solver result against oracle
struct Verdict {
    bool pass;
    std::string reason;
};

Verdict verify(const TestInstance& inst, const espprc::Result& res) {
    bool oracle_has_neg = (inst.oracle_rc < -1e-6);
    bool solver_has_neg = (res.optimal_rc < 1e29);

    if (oracle_has_neg) {
        // Oracle says there exists a negative-RC route
        if (!solver_has_neg)
            return {false, "solver found no route, oracle RC=" +
                           std::to_string(inst.oracle_rc)};

        // For proven-optimal oracles (enumerate/mip), RC should match
        // For time_limit oracles, solver may find BETTER (more negative) RC
        bool oracle_exact = (inst.oracle_method.find("time_limit") == std::string::npos);

        if (oracle_exact) {
            double gap = std::abs(res.optimal_rc - inst.oracle_rc);
            if (gap > TOL)
                return {false, "RC mismatch: solver=" + std::to_string(res.optimal_rc) +
                               " oracle=" + std::to_string(inst.oracle_rc)};
        } else {
            // Oracle is upper bound; solver should find RC <= oracle RC
            if (res.optimal_rc > inst.oracle_rc + TOL)
                return {false, "solver RC=" + std::to_string(res.optimal_rc) +
                               " > oracle UB=" + std::to_string(inst.oracle_rc)};
        }
    } else {
        // Oracle says no negative-RC route (converged)
        if (solver_has_neg && res.optimal_rc < -TOL)
            return {false, "solver found neg RC=" + std::to_string(res.optimal_rc) +
                           " but oracle says RC>=0"};
    }

    // Verify path feasibility
    if (solver_has_neg && !res.optimal_path.empty()) {
        // Check elementary
        std::vector<bool> seen(inst.n_customers + 1, false);
        for (int v : res.optimal_path) {
            if (v == 0) continue;
            if (seen[v])
                return {false, "path not elementary: repeated node " + std::to_string(v)};
            seen[v] = true;
        }
        // Check capacity
        int load = 0;
        for (int v : res.optimal_path)
            if (v != 0) load += inst.demand[v];
        if (load > inst.capacity)
            return {false, "capacity violated: load=" + std::to_string(load) +
                           " > " + std::to_string(inst.capacity)};

        // Check path starts and ends at depot
        if (res.optimal_path.front() != 0 || res.optimal_path.back() != 0)
            return {false, "path doesn't start/end at depot"};
    }

    return {true, ""};
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string inst_dir;
    int max_n = 9999;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--up-to" && i + 1 < argc)
            max_n = std::stoi(argv[++i]);
        else
            inst_dir = arg;
    }

    // Default instance directory: relative to executable or project root
    if (inst_dir.empty()) {
        // Try common locations
        for (auto& candidate : {
            "studies/espprc_testset/instances",
            "../../../studies/espprc_testset/instances",
            "../../../../studies/espprc_testset/instances",
        }) {
            if (fs::exists(candidate)) { inst_dir = candidate; break; }
        }
    }
    if (inst_dir.empty() || !fs::exists(inst_dir)) {
        std::cerr << "Instance directory not found. Usage: test_espprc [path] [--up-to N]\n";
        return 1;
    }

    // Collect JSON files
    std::vector<fs::path> files;
    for (auto& entry : fs::directory_iterator(inst_dir)) {
        auto fn = entry.path().filename().string();
        if (fn.size() > 5 && fn.substr(fn.size() - 5) == ".json"
            && fn != "manifest.json")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    std::cout << "ESPPRC Test Runner\n";
    std::cout << std::string(72, '=') << "\n";
    std::cout << "Instances: " << inst_dir << "  (" << files.size() << " files)\n";
    if (max_n < 9999)
        std::cout << "Filter: n <= " << max_n << "\n";
    std::cout << std::string(72, '-') << "\n";

    // Header
    std::cout << std::left << std::setw(25) << "Instance"
              << std::right << std::setw(4)  << "n"
              << std::setw(12) << "Oracle RC"
              << std::setw(12) << "Solver RC"
              << std::setw(10) << "Labels"
              << std::setw(10) << "Time(ms)"
              << "  Result\n";
    std::cout << std::string(72, '-') << "\n";

    int pass_count = 0, fail_count = 0, skip_count = 0;

    for (auto& filepath : files) {
        TestInstance inst = load_instance(filepath);

        if (inst.n_customers > max_n) {
            skip_count++;
            continue;
        }

        // Run solver (30s timeout per instance)
        espprc::Result res = espprc::solve(
            inst.dist, inst.demand, inst.capacity, inst.cover_duals, 30000.0);

        // Verify
        Verdict v = {true, ""};
        if (res.timed_out)
            v = {false, "TIMEOUT"};
        else
            v = verify(inst, res);

        // Format output
        char buf_oracle[32], buf_solver[32];
        if (inst.oracle_rc > 1e20)
            snprintf(buf_oracle, sizeof(buf_oracle), "%s", "N/A");
        else
            snprintf(buf_oracle, sizeof(buf_oracle), "%.2f", inst.oracle_rc);

        if (res.optimal_rc > 1e20)
            snprintf(buf_solver, sizeof(buf_solver), "%s", "none");
        else
            snprintf(buf_solver, sizeof(buf_solver), "%.2f", res.optimal_rc);

        std::cout << std::left << std::setw(25) << inst.name
                  << std::right << std::setw(4) << inst.n_customers
                  << std::setw(12) << buf_oracle
                  << std::setw(12) << buf_solver
                  << std::setw(10) << res.labels_created
                  << std::setw(10) << std::fixed << std::setprecision(1) << res.time_ms
                  << "  " << (v.pass ? "PASS" : "FAIL");
        if (!v.pass)
            std::cout << " (" << v.reason << ")";
        std::cout << std::endl;   // flush each line

        if (v.pass) pass_count++;
        else        fail_count++;
    }

    std::cout << std::string(72, '-') << "\n";
    std::cout << "Total: " << pass_count << " passed, "
              << fail_count << " failed, "
              << skip_count << " skipped\n";

    return fail_count > 0 ? 1 : 0;
}
