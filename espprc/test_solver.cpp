// test_solver.cpp — Verify espprc_solver.h unified interface
// Build: cmake --build . --config Release --target test_solver
// Run:   ./Release/test_solver.exe

#include "espprc_solver.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    // T3 benchmark instance: 3 customers, known optimal
    std::vector<std::vector<int>> dist = {
        {0, 3, 4, 5},
        {3, 0, 5, 4},
        {4, 5, 0, 3},
        {5, 4, 3, 0}
    };
    std::vector<int> demand = {0, 2, 3, 2};
    int capacity = 4;
    std::vector<double> duals = {6.0, 8.0, 6.0};  // converged duals

    // At converged duals, no negative RC path should exist
    espprc::Solver solvers[] = {
        espprc::Solver::NG_POOL,
        espprc::Solver::BIDIR_POOL,
        espprc::Solver::EXACT_POOL,
        espprc::Solver::DSSR_POOL,
    };

    std::cout << "=== Converged duals (expect no negative RC) ===" << std::endl;
    for (auto s : solvers) {
        auto r = espprc::solve(dist, demand, capacity, duals, s);
        std::cout << espprc::solver_name(s)
                  << ": rc=" << r.optimal_rc
                  << " neg=" << r.has_negative_rc()
                  << " time=" << r.time_ms << "ms"
                  << " labels=" << r.labels_total
                  << std::endl;
        assert(!r.has_negative_rc());
    }

    // Large duals: should find negative RC paths
    // Path 0->1->3->0: dist=3+4+5=12, duals=10+10=20, rc=-8
    std::vector<double> large_duals = {10.0, 10.0, 10.0};
    std::cout << "\n=== Large duals (expect negative RC) ===" << std::endl;
    for (auto s : solvers) {
        auto r = espprc::solve(dist, demand, capacity, large_duals, s);
        std::cout << espprc::solver_name(s)
                  << ": rc=" << r.optimal_rc
                  << " path=[";
        for (int i = 0; i < (int)r.optimal_path.size(); i++) {
            if (i) std::cout << ",";
            std::cout << r.optimal_path[i];
        }
        std::cout << "] elem=" << r.path_is_elementary
                  << " time=" << r.time_ms << "ms"
                  << std::endl;
        assert(r.has_negative_rc());
        assert(r.optimal_path.front() == 0);
        assert(r.optimal_path.back() == 0);
    }

    // Verify all exact solvers agree on RC
    auto r_exact = espprc::solve(dist, demand, capacity, large_duals, espprc::Solver::EXACT_POOL);
    auto r_dssr  = espprc::solve(dist, demand, capacity, large_duals, espprc::Solver::DSSR_POOL);
    assert(std::abs(r_exact.optimal_rc - r_dssr.optimal_rc) < 1e-6);

    std::cout << "\nAll tests passed." << std::endl;
    return 0;
}
