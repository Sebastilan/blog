cmake_minimum_required(VERSION 3.16)
project(espprc_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# No LP solver dependency — pure algorithmic code
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/relaxed ${CMAKE_SOURCE_DIR}/exact)

add_executable(test_espprc test_espprc.cpp)
add_executable(bench_espprc bench_espprc.cpp)
add_executable(bench_optimization bench_optimization.cpp)
add_executable(solve_one solve_one.cpp)
add_executable(test_solver test_solver.cpp)
add_executable(bench_exact_ablation bench_exact_ablation.cpp)
add_executable(bench_exact_ablation_quick bench_exact_ablation_quick.cpp)
add_executable(bench_exact_final bench_exact_final.cpp)

# k-cycle elimination benchmark
add_executable(bench_kcycle bench_kcycle.cpp)

# Micro-optimization A/B tests (4 variants of bench_optimization)
add_executable(bench_micro_baseline bench_optimization.cpp)

add_executable(bench_micro_binlimit bench_optimization.cpp)
target_compile_definitions(bench_micro_binlimit PRIVATE ESPPRC_OPT_BIN_LIMIT=128)

add_executable(bench_micro_adaptive bench_optimization.cpp)
target_compile_definitions(bench_micro_adaptive PRIVATE ESPPRC_OPT_ADAPTIVE_MEET=1)

add_executable(bench_micro_distsort bench_optimization.cpp)
target_compile_definitions(bench_micro_distsort PRIVATE ESPPRC_OPT_DIST_SORT=1)

# Release by default for meaningful timing
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
