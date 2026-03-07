// solve_one.cpp — 单实例求解，输出 JSON 格式结果
// 用法: solve_one <instance.json> [--solver ng|exact|dssr] [--delta D] [--timelimit MS]
// 输出一行 JSON: {"solver":"...","rc":...,"time_ms":...,"path":[...],"elementary":...}

#include "relaxed/espprc_ng_bidir_pool.h"
#include "exact/espprc_exact_pool.h"
#include "exact/espprc_dssr_pool.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: solve_one <instance.json> [--solver ng|exact|dssr] [--delta D] [--timelimit MS]\n";
        return 1;
    }

    std::string filepath = argv[1];
    std::string solver_name = "all";  // default: run all
    int delta = 8;
    double time_limit = 60000.0;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--solver" && i + 1 < argc) solver_name = argv[++i];
        else if (arg == "--delta" && i + 1 < argc) delta = std::stoi(argv[++i]);
        else if (arg == "--timelimit" && i + 1 < argc) time_limit = std::stod(argv[++i]);
    }

    // Load instance
    std::ifstream ifs(filepath);
    if (!ifs) { std::cerr << "Cannot open " << filepath << "\n"; return 1; }
    json j = json::parse(ifs);

    int n_cust = j["n_customers"];
    int N = n_cust + 1;
    int capacity = j["capacity"];

    std::vector<std::vector<int>> dist(N, std::vector<int>(N));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            dist[i][k] = j["dist"][i][k].get<int>();

    std::vector<int> demand(N);
    for (int i = 0; i < N; i++)
        demand[i] = j["demand"][i].get<int>();

    std::vector<double> duals(n_cust);
    for (int i = 0; i < n_cust; i++)
        duals[i] = j["cover_duals"][i].get<double>();

    // Output JSON array
    json results = json::array();

    // ng_bidir_pool (L2 bidirectional + pool)
    if (solver_name == "all" || solver_name == "bidir") {
        auto r = espprc_ng_bidir_pool::solve(dist, demand, capacity, duals, delta, 20, time_limit);
        results.push_back({
            {"solver", "bidir_pool_d" + std::to_string(delta)},
            {"rc", r.optimal_rc < 1e29 ? json(r.optimal_rc) : json(nullptr)},
            {"time_ms", r.time_ms},
            {"path", r.optimal_path},
            {"elementary", r.path_is_elementary},
            {"fwd_labels", r.fwd_labels},
            {"bwd_labels", r.bwd_labels},
            {"arcs_eliminated", r.arcs_eliminated},
            {"arcs_total", r.arcs_total},
            {"labels_pruned", r.labels_pruned},
            {"timed_out", r.timed_out}
        });
    }

    // exact_pool (L3 full bitmask)
    if (solver_name == "all" || solver_name == "exact") {
        auto r = espprc_exact_pool::solve(dist, demand, capacity, duals, 20, time_limit);
        results.push_back({
            {"solver", "exact_pool"},
            {"rc", r.optimal_rc < 1e29 ? json(r.optimal_rc) : json(nullptr)},
            {"time_ms", r.time_ms},
            {"path", r.optimal_path},
            {"elementary", r.path_is_elementary},
            {"labels", r.labels_created},
            {"dom_checks", r.dom_checks},
            {"timed_out", r.timed_out}
        });
    }

    // dssr_pool (L3 DSSR)
    if (solver_name == "all" || solver_name == "dssr") {
        auto r = espprc_dssr_pool::solve(dist, demand, capacity, duals, delta, 20, 20, time_limit);
        results.push_back({
            {"solver", "dssr_pool_d" + std::to_string(delta)},
            {"rc", r.optimal_rc < 1e29 ? json(r.optimal_rc) : json(nullptr)},
            {"time_ms", r.time_ms},
            {"path", r.optimal_path},
            {"elementary", r.path_is_elementary},
            {"labels", r.labels_created},
            {"timed_out", r.timed_out},
            {"dssr_iterations", r.dssr_iterations},
            {"critical_count", r.critical_count}
        });
    }

    std::cout << results.dump() << std::endl;
    return 0;
}
