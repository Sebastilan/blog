# ESPPRC Solver Library

CVRP 子问题（ESPPRC / SPPRC）C++ 求解器库，用于 Branch-Price-and-Cut 列生成的定价子问题。

## 目录结构

```
espprc/
├── exact/                          # 精确求解器
│   ├── espprc.h                    #   全 bitmask oracle
│   ├── espprc_exact_pool.h         #   Full Bitmask + Pool (L3 备选)
│   ├── espprc_dssr_pool.h          #   DSSR + Pool (L3 首选)
│   ├── espprc_exact_modular.h      #   模块化精确求解器（消融实验用）
│   └── espprc_exact_ablation.h     #   C0/C1/C2 消融变体
├── relaxed/                        # 松弛求解器
│   ├── espprc_ng_bidir_pool.h      #   ng-route + 双向 + 并行 + Pool (默认推荐)
│   ├── espprc_ng.h                 #   ng-mask 构建工具
│   └── espprc_kcycle.h             #   k-cycle 消除（对比实验用）
├── docs/                           # 文档 + 实验报告
│   ├── strategies.md               #   策略研究总览（13 个求解器 + 性能对比）
│   ├── exact_ablation.md           #   精确型策略消融实验（Pool/Bucket/Bidir/Parallel）
│   └── complexity.md               #   T_pricing 复杂度分解 + 策略→因子映射
├── testset/                        # 测试数据（JSON，含 CG 真实对偶值）
├── _archive/                       # 历史版本
├── espprc_solver.h                 # 统一接口
├── bench_exact_ablation.cpp        # 消融 benchmark（9 配置 × 4 实例）
├── bench_espprc.cpp                # 综合 benchmark
├── bench_kcycle.cpp                # k-cycle vs ng-route 对比 benchmark
├── bench_optimization.cpp          # 微优化 A/B 对比 benchmark
└── README.md                       # 本文件
```

## 求解器一览

| 求解器 | 类型 | 算法 | 适用场景 |
|--------|------|------|----------|
| `BIDIR_POOL` | 松弛 (L2) | ng-route + 双向并行 + Pool | **默认推荐**，大实例最快 |
| `EXACT_POOL` | 精确 (L3) | 全 bitmask + Pool | n ≤ 20，需要精确解 |
| `DSSR_POOL` | 精确 (L3) | DSSR 迭代收紧 + Pool | n ≤ 31，需要精确解 |

### 性能参考（vs cspy，学术界主流开源求解器）

| 对比 | 加速倍数 |
|------|----------|
| 松弛：BIDIR_POOL vs cspy SPPRC | 100–1,000x |
| 精确：EXACT_POOL vs cspy elementary | 100–31,000x |

### 内部加速技术

- **Pool 数据结构**：预分配标号池 + RC 排序桶 + min-RC 跳过
- **双向搜索**：fwd (load ≤ Q/2) + bwd (load ≤ Q−Q/2) + 合并
- **并行**：fwd/bwd 双线程 (n ≥ 15 时自动启用)
- **Merge 优化**：pair-level 下界跳过 + RC 排序双重早停
- **ng-route 松弛**：O(n·2^Δ) 状态空间，Δ = 8 时远小于 O(n·2^n)
- **DSSR**：逐轮识别 critical vertices，逐步收紧到精确解
- **DistSort**：按距离排序邻居扩展顺序，1.2-1.5x 加速（默认启用）

## 快速开始

### 统一接口

```cpp
#include "espprc_solver.h"

// 输入
std::vector<std::vector<int>> dist = {...};  // N×N 距离矩阵, N = n_customers + 1
std::vector<int> demand = {...};              // 需求, demand[0] = 0 (depot)
int capacity = 100;
std::vector<double> duals = {...};            // 覆盖约束对偶值 pi[1..n_cust]

// 求解（默认 BIDIR_POOL）
auto r = espprc::solve(dist, demand, capacity, duals);

if (r.has_negative_rc()) {
    // r.optimal_path = [0, v1, v2, ..., vk, 0]
    // r.optimal_rc   = 最小 reduced cost
    // 将 r.optimal_path 作为新列加入 RMP
}
```

### 指定求解器和参数

```cpp
espprc::Config cfg;
cfg.delta = 12;          // ng 邻域大小（越大越紧，越慢）
cfg.theta = 20;          // 桶粒度
cfg.time_limit_ms = 5000; // 5 秒超时

auto r = espprc::solve(dist, demand, capacity, duals,
                       espprc::Solver::DSSR_POOL, cfg);
```

### 直接调用底层求解器

```cpp
#include "relaxed/espprc_ng_bidir_pool.h"

auto r = espprc_ng_bidir_pool::solve(dist, demand, capacity, duals,
                                      /*delta=*/8, /*theta=*/20, /*time_limit=*/60000);
```

## 输入格式

| 参数 | 类型 | 说明 |
|------|------|------|
| `dist` | `vector<vector<int>>` | N×N 整数距离矩阵，N = n_customers + 1 |
| `demand` | `vector<int>` | 长度 N，demand[0] = 0（depot） |
| `capacity` | `int` | 车辆容量 Q |
| `cover_duals` | `vector<double>` | 长度 n_customers，覆盖约束对偶值 π |

**Reduced cost 计算**：`rc(i,j) = dist[i][j] − π[j]`（π[0] = 0）

## 输出字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `optimal_rc` | `double` | 最优 reduced cost（1e30 = 无路径） |
| `optimal_cost` | `int` | 最优路径的原始距离成本 |
| `optimal_path` | `vector<int>` | 节点序列 [0, v1, ..., vk, 0] |
| `path_is_elementary` | `bool` | 路径是否无重复客户 |
| `has_negative_rc()` | `bool` | `optimal_rc < -1e-6` |
| `timed_out` | `bool` | 是否超时 |
| `time_ms` | `double` | 求解耗时（毫秒） |
| `labels_total` | `int` | 创建的标号总数 |
| `dom_checks` | `long long` | 支配性检查次数 |

## 编译

依赖：C++17，nlohmann/json（仅 solve_one.cpp 需要）。

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## 文件结构

```
espprc/
├── espprc_solver.h              # 统一接口（推荐使用）
├── relaxed/
│   ├── espprc_ng_bidir_pool.h   # ng-route 松弛，双向并行，Pool ★
│   ├── espprc_ng.h              # ng-mask 构建工具
│   └── espprc_kcycle.h          # k-cycle 消除（对比实验用）
├── exact/
│   ├── espprc.h                 # 精确 ESPPRC oracle
│   ├── espprc_dssr_pool.h       # 精确 ESPPRC，DSSR 迭代，Pool
│   └── espprc_exact_pool.h      # 精确 ESPPRC，全 bitmask，Pool
├── _archive/                    # 历史版本（9 个）
├── solve_one.cpp                # CLI 单实例求解工具
├── bench_espprc.cpp             # 综合 benchmark
├── bench_optimization.cpp       # 优化策略 A/B 对比 benchmark
├── bench_kcycle.cpp             # k-cycle vs ng-route 对比 benchmark
├── test_espprc.cpp              # 正确性测试
├── test_solver.cpp              # 统一接口测试
├── CMakeLists.txt
└── README.md                    # 本文件
```

## 优化实验记录（2026-03）

对 BIDIR_POOL 进行了全面的微优化 A/B 测试：

| 策略 | 结论 | 原因 |
|------|------|------|
| **DistSort（距离排序扩展）** | **✅ 有效，默认启用** | 近邻先扩展，1.2-1.5x 加速，标号减少 12-35% |
| Label-per-bin limit (128) | ❌ 无效 | 测试规模标号数不足以触发桶溢出 |
| Adaptive meet-point | ❌ 无效 | CVRP 单资源，fwd/bwd 天然平衡 |
| Reachable Bitmask | ❌ 无效 | ng-mask 稀疏，bitwise 无优势 |
| Greedy 初始上界 | ❌ 无效 | merge 自然收紧极快 |
| Jump Arcs (Sadykov 2021) | ❌ 不适用 | 需 2+ 资源维度 |

### k-cycle vs ng-route 对比实验

实现了 k-cycle 消除求解器（`espprc_kcycle.h`），验证 ng-route 的空间邻域优于 k-cycle 的时序间隔：
- k-cycle K=3 在 n=31 需 33s，ng Δ=8 仅 86ms（385x 差距）
- k-cycle bound 质量更差（n=15: RC=-196 vs ng/exact RC=-131）
- 详见 `bench_kcycle.cpp` 和 `docs/strategies.md`

实验代码保留：`bench_optimization.cpp`（A/B 对比），`bench_kcycle.cpp`（k-cycle 对比）。DistSort 默认开启（`ESPPRC_OPT_DIST_SORT=1`），其余开关默认关闭。

## 限制

- **n_customers ≤ 63**：使用 uint64_t bitmask 表示访问集合
- **整数距离**：输入距离矩阵为 int 类型（CVRPLIB 标准）
- **单一资源**：仅支持容量约束（无时间窗）

## 技术对标

本库的技术栈对齐 RouteOpt 2.0（2025 INFORMS Journal on Computing）的单次定价模块：

| 技术 | RouteOpt 2.0 | 本库 |
|------|-------------|------|
| ng-route 松弛 | ✓ | ✓ (BIDIR_POOL) |
| 双向标号 | ✓ | ✓ (BIDIR_POOL) |
| Pool / Bucket Graph | ✓ | ✓ (所有求解器) |
| DSSR 精确化 | ✓ | ✓ (DSSR_POOL) |
| 并行 fwd/bwd | ✓ | ✓ (BIDIR_POOL, n≥15) |
| RC 排序 + 早停 | ✓ | ✓ (Merge 阶段) |
