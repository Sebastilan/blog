# Dantzig & Ramser (1959) — 逐层捆绑算法复现

> G.B. Dantzig, J.H. Ramser. "The Truck Dispatching Problem." *Management Science*, 6(1):80–91, 1959.

VRP（Vehicle Routing Problem）的开山之作。本目录包含论文算法的完整复现代码和标准 CVRP 测试数据。

## 文件说明

| 文件 | 说明 |
|------|------|
| `dantzig_example.py` | 论文 12 站数值示例专用代码（自包含，Gurobi） |
| `dantzig_solver.py` | 通用 CVRP 求解器（支持标准 .vrp 格式，Gurobi + LKH） |
| `benchmarks/` | 115 个 CVRPLIB 标准实例 + 已知最优解 |

## 依赖

```bash
pip install numpy gurobipy vrplib elkai
```

- **Gurobi**: 需要许可证（学术免费）
- **elkai**: LKH 启发式 TSP 求解器的 Python 封装

## 运行

```bash
# 论文 12 站示例（输出总距离 294）
python dantzig_example.py

# 求解单个实例
python dantzig_solver.py benchmarks/cvrp/E-n13-k4.vrp

# 跑全部 benchmark（n <= 50，共 49 个实例）
python dantzig_solver.py
```

## Benchmark 结果

49 个实例（客户数 <= 50），平均 Gap 17.8%，最佳 0.2%（E-n23-k3），最差 29.9%（B-n43-k6）。

详见 [CSDN 博客](https://blog.csdn.net/fair_li/article/details/158097337)。

## 算法简述

逐层捆绑：每一层求解匹配 LP，把节点两两配对捆绑为更大的组，绑定后不可拆，交给下一层继续合并。层数 L = ceil(log2(t))，t 为每车最多站数。
