# Gurobi vs COPT：分支定价（Branch-and-Price）场景下的求解器性能实测

> 在 BPC 求解框架中，商业求解器承担着 LP 松弛、列生成、MIP 启发式等核心角色。Gurobi 和 COPT 是目前最主流的两个选择。本文从 BPC 的实际操作出发，设计 8 项针对性测试，用数据说明两者的差异。

## 1 背景与动机

分支定价（Branch-and-Price, B&P）是求解 CVRP 等组合优化问题的精确算法。在 B&P 框架中，商业 LP/MIP 求解器并非"调一次 `solve()`"那么简单——它需要在整个求解过程中反复执行一系列**细粒度操作**：

- 每轮列生成（CG）：求解 LP、提取对偶、添加新列
- 每个分支节点：修改变量 bound、增删约束、warm start LP
- 启发式阶段：求解 Restricted MIP

因此，选择求解器时不能只看"解一个大 LP 多快"，还要看**模型修改的 API 效率**和 **warm start 的加速效果**。

本文对比 **Gurobi 13.0** 和 **COPT 8.0.3**（杉数求解器），在我们从零搭建的 BPC 求解框架上进行实测。

## 2 测试设计

### 2.1 测试环境

| 项目 | 配置 |
|------|------|
| CPU | Intel Core i5-12500H (12C/16T) |
| OS | Windows 11 |
| Python | 3.12 |
| Gurobi | 13.0.0（学术许可） |
| COPT | 8.0.3（学术试用 180 天） |
| 线程数 | 固定为 1（公平比较） |

### 2.2 基准实例

使用 CVRPLIB 的 **P-n16-k8** 实例（15 个客户节点，容量约束 CVRP）：

- 通过列生成构建 **412 条路径列**（column pool）
- Root LP 最优值 = 441.00（两个求解器一致）
- 这个规模代表了 BPC 中一个典型的 RMP（Restricted Master Problem）

### 2.3 测试项目

从 BPC 的实际操作流程出发，设计 8 项测试：

| 编号 | 测试项 | B&P 中的场景 | 测试方法 |
|------|--------|-------------|---------|
| T1 | LP 求解速度 | 每轮 CG 核心操作 | 同一模型重复求解 60 次 |
| T2 | 批量添加列 | CG 添加负 reduced cost 列 | 分别加 50/200/412 列 |
| T3 | 批量增删约束 | 分支 forced edge / cuts | 加 10-100 条约束 + 求解 + 删除 + 求解 |
| T4 | 变量 bound 修改 | 分支 forbidden edge | 206 个变量 UB 切换 0/∞，30 轮 |
| T5 | 对偶值提取 | CG 每轮提取 π 传给定价子问题 | 15 个对偶值 × 1500 次 |
| T6 | Basis warm start | CG 相邻轮次热启动 | 有/无 basis 的 LP 求解对比 |
| T7 | MIP 求解 | Restricted MIP 启发式 | 412 个 binary 变量的 set-covering IP |
| T8 | 端对端 CG | 完整列生成流程 | Root CG 收敛全程计时 |

每项测试跑 3 轮取中位数，确保数据稳定。

## 3 实验结果

### 3.1 总览

| 测试项 | Gurobi | COPT | COPT/Gurobi | 判定 |
|--------|--------|------|:-----------:|:----:|
| T1: LP 求解 (×60) | 0.0017s | 0.0051s | **3-4x** | Gurobi 胜 |
| T2: 加 412 列 | 0.0020s | 0.0016s | **~1x** | 持平 |
| T3: 加 100 约束+求解 | 0.0012s | 0.0016s | **~1.4x** | 基本持平 |
| T3: 删 100 约束+求解 | 0.0002s | 0.0005s | **~2-3x** | Gurobi 略优 |
| T4: 变量 bound 切换 (×30) | 0.0188s | 0.0146s | **~1x** | 持平 |
| T5: 对偶提取 (×1500) | 0.0188s | 0.0157s | **0.84x** | COPT 略优 |
| T6: Cold start LP (×60) | 0.0275s | 0.1563s | **5-6x** | Gurobi 明显快 |
| T6: Warm start LP (×60) | 0.0004s | 0.0052s | **10-14x** | **Gurobi 大幅领先** |
| T7: MIP 求解 | 0.0066s | 0.4001s | **8-60x** | **Gurobi 碾压** |
| T8: Root CG 端对端 | 0.012s | 0.014s | **~1x** | 持平 |

### 3.2 分项分析

#### T1: LP 求解——Gurobi Simplex 内核更快

60 次重复求解同一个 15×412 的 LP（模型不变，测纯 simplex 性能）：

```
Gurobi:  0.0017s (60 solves)
COPT:    0.0051s (60 solves)
Ratio:   COPT / Gurobi ≈ 3x
```

对于 BPC 的 RMP（行少列多的典型退化 LP），Gurobi 的 dual simplex 有明显优势。

#### T2: 批量添加列——基本持平

```
50 列:   Gurobi 0.0004s,  COPT 0.0002s  → COPT 略快
200 列:  Gurobi 0.0011s,  COPT 0.0016s  → 基本持平
412 列:  Gurobi 0.0020s,  COPT 0.0016s  → 基本持平
```

Column API（构造列对象 + addVar）的开销两者相当，不是性能瓶颈。

#### T3: 批量增删约束——Gurobi 删约束更快

```
加 100 约束 + 求解:   Gurobi 0.0012s,  COPT 0.0016s  (1.4x)
删 100 约束 + 求解:   Gurobi 0.0002s,  COPT 0.0005s  (2-3x)
```

这对 Phase 2 的 cuts（R1C、RCC）有影响——切割平面需要频繁增删。Gurobi 在约束删除后重新求解的效率更高。

#### T4 & T5: 变量 bound 修改与对偶提取——持平

```
T4 (bound 切换 ×30):     Gurobi 0.0188s,  COPT 0.0146s  → 持平
T5 (15 duals ×1500):     Gurobi 0.0188s,  COPT 0.0157s  → COPT 略优
```

两者在纯数据操作（修改 bound、读取 dual）上没有显著差异。

**注意**：T5 中两个求解器返回的对偶值**不相同**（max diff = 2.67）。这是 LP 退化（degeneracy）导致的——最优解不唯一时，不同 simplex 实现选择不同的基，对偶值也不同。这会导致 CG 走不同路径（生成不同的列），最终搜索树可能完全不同。

#### T6: Basis Warm Start——差距最大的环节

```
Cold start (×60):   Gurobi 0.0275s,  COPT 0.1563s  →  5-6x
Warm start (×60):   Gurobi 0.0004s,  COPT 0.0052s  → 10-14x
```

**这是对 BPC 性能影响最大的一项。** CG 的核心循环是：加几列 → 重新求解 LP → 提取对偶 → 定价。每次重新求解时，模型只有微小变化（多了几个变量），理论上只需要很少的 simplex 迭代。Gurobi 的 warm start 将这个过程压缩到了极致（0.007ms/次），而 COPT 需要约 0.087ms/次。

在一个有 100 轮 CG 迭代的节点中，仅 LP warm start 的累计差异就是 0.7ms vs 8.7ms。虽然绝对值不大，但在大规模实例上（数千个 B&B 节点 × 数十轮 CG），这个 10x 的差距会被放大。

#### T7: MIP 求解——Gurobi 碾压

```
MIP (412 binary vars):  Gurobi 0.007s,  COPT 0.400s  → 8-60x
```

两者都找到了最优解 450，但 Gurobi 的 Branch-and-Cut 引擎快了 1-2 个数量级。这影响 Restricted MIP 启发式的效率——BPC 在每个节点用当前列池求一个 IP 来找上界。

不过 MIP 求解时间波动较大（3 轮分别为 8x、60x、147x），可能与 Gurobi 的启发式搜索策略有关。

#### T8: 端对端 Root CG——定价主导，求解器差异被淹没

```
Gurobi:  0.012s - 0.060s
COPT:    0.014s - 0.054s
Ratio:   ~1x
```

在完整的列生成流程中（P-n16-k8，2 轮 CG），两个求解器的总时间几乎一样。原因是**定价子问题（ESPPRC）占了 95% 以上的时间**——这是我们自己的 Python label-setting 算法，跟求解器无关。

这说明：在当前的基础 BPC 实现中，**加速定价子问题比换求解器更重要**。但当我们在 Phase 2 实现了更快的定价（ng-route、bucket graph），RMP 的 LP 求解时间占比会上升，届时 Gurobi 的 warm start 优势会更加显著。

## 4 对偶退化的"蝴蝶效应"

T5 揭示了一个有趣的现象：同一个 LP 的最优解，两个求解器返回了不同的对偶值（max diff = 2.67）。这不是 bug，而是 LP 退化的必然结果——当约束矩阵退化时，最优基不唯一，不同 simplex 路径选择不同的基。

在 BPC 中，对偶值直接决定了定价子问题的 reduced cost，进而决定了哪些列被生成。**同一个实例，Gurobi 和 COPT 可能走完全不同的 CG 路径，探索完全不同的搜索树。** 我们在 Phase 1 的开发中已经观察到过这个现象——同一实例 Python（Gurobi）vs C++（Gurobi）因为浮点差异导致 6 轮 vs 1507 轮 CG 迭代。

这意味着：**比较求解器在 B&P 端对端上的表现时，不能只看一个实例——需要大量实例取统计平均。**

## 5 结论与建议

### 5.1 量化总结

将 8 项测试按对 BPC 性能的影响权重排序：

| 优先级 | 操作 | Gurobi 优势 | 影响程度 |
|:------:|------|:-----------:|:--------:|
| 1 | Warm start LP（CG 核心循环） | **10-14x** | 极高 |
| 2 | Cold start LP（首次求解） | **5-6x** | 高 |
| 3 | MIP 求解（启发式上界） | **8-60x** | 中 |
| 4 | 约束删除 + 重求解（cuts 管理） | **2-3x** | 中 |
| 5 | 纯 LP 求解（simplex 内核） | **3-4x** | 中 |
| 6 | 加列 / 改 bound / 读对偶 | **~1x** | 低（持平） |

### 5.2 选择建议

- **追求极致性能**：Gurobi。warm start 和 MIP 求解的优势在大规模 BPC 中会成倍放大。
- **国产化 / 成本敏感**：COPT 完全可用。在模型操作层面与 Gurobi 持平，主要差距在 simplex 内核和 MIP 引擎——对于定价主导的场景，这个差距可能不是瓶颈。
- **学术研究**：两者都提供免费学术许可。如果同时有，建议用 Gurobi 做主力、COPT 做验证。

### 5.3 对我们 BPC 项目的启示

1. **当前阶段**（基础 BPC，Python 定价）：求解器差异被定价淹没，用哪个都行
2. **Phase 2**（加速定价后）：RMP LP 求解时间占比将上升，Gurobi 的 warm start 优势会显现
3. **C++ 实现**：已有 `solver_gurobi.h` / `solver_copt.h` 双后端，随时可以切换验证

## 6 复现指南

### 6.1 环境要求

- Python 3.8+
- `gurobipy`（需 Gurobi 许可）
- `coptpy`（需 COPT 许可）
- `vrplib`、`numpy`

### 6.2 运行方式

```bash
cd studies/solver_comparison

# 快速验证（~10s）
python -X utf8 benchmark.py --quick

# 完整测试（~30s）
python -X utf8 benchmark.py

# 3 轮取中位数（~90s）
python -X utf8 benchmark.py --rounds 3
```

### 6.3 代码结构

```
solver-comparison/
├── benchmark.py      # 主测试脚本（T1-T8）
├── rmp.py            # Gurobi 版 RMP
├── rmp_copt.py       # COPT 版 RMP（接口一致）
├── instance.py       # CVRP 实例加载
├── pricing.py        # ESPPRC 定价子问题
├── branch.py         # 边分支
├── utils.py          # 辅助工具
├── blog.md           # 本文
├── data/             # 测试实例
└── results/          # 实验数据（JSON）
```

完整代码和数据已开源在 GitHub：[solver-comparison](https://github.com/Sebastilan/blog/tree/master/vrp-papers/solver-comparison)

---

*测试日期: 2026-03-02 | 作者: Sebastian Li*
