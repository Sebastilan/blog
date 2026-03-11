# 架构：怎么组织的？

## 关键文件/目录

| 路径 | 作用 |
|------|------|
| README.md | 文章索引 |
| vrp-papers/ | VRP 论文笔记系列 |
| vrp-papers/dantzig-1959/ | Dantzig 1959 经典论文复现 |
| vrp-papers/cvrp-mip/ | CVRP 六种 MIP 模型对比 |
| vrp-papers/solver-comparison/ | Gurobi vs COPT 性能实测 |
| vrp-papers/engineering-acceleration/ | BPC 工程加速 5 策略 A/B 测试 |
| ai-benchmarks/ | AI 模型评测 |
| ai-benchmarks/2026-02-12-deepseek-vs-qwen/ | 四模型 Function Calling 实测 |
| espprc/ | ESPPRC 求解器库（C++，独立文章） |

## 文章结构

每篇文章一个目录，包含：
- `README.md` 或 `blog.md`（正文）
- Python/C++ 代码（可复现实验）
- 数据文件（CVRPLIB 实例、测试结果）

## 依赖

- Gurobi（MIP 实验）
- CVRPLIB 标准实例
- elkai、vrplib（Python 包）
