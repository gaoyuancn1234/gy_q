# QuantaAlpha: Multi-Agent Quant Factor Mining

> 论文: [QuantaAlpha: An AI-Driven Multi-Agent Framework for Automated Quantitative Factor Mining (2602.07085)](2602.07085v1.pdf)

## 论文摘要

QuantaAlpha 提出了一个多 Agent 协作的量化因子挖掘框架，核心流程为 5 步演化循环：
Propose → Construct → Calculate → Backtest → Feedback，
通过自我演化 (Mutation/Crossover) 逐步改进因子质量。

## 复现方法

```bash
# 环境
conda activate quantaalpha  # Python 3.10

# 论文规模复现 (10方向, 5轮)
python -m factor_lab.run_quanta_alpha

# 9窗口 walk-forward rolling 评估
python scripts/run_paper_replication.py --eval-only
```

**Rolling Runner**: `quantaalpha/factors/rolling_runner.py` — H5 注入 + 9 窗口 walk-forward

**注意**: QuantaAlpha 表达式 ≠ Qlib 表达式，必须读 result.h5 注入 bin 文件。

## 关键发现

### 1. 评估方式决定因子质量
- 旧单次训练因子：Sharpe **-22%** (劣化)
- 新 rolling 因子：Sharpe **+17%** (提升)
- **结论**: 因子必须在 rolling walk-forward 下评估才能反映真实质量

### 2. Window 8 崩溃根因
- 验证集 Q3 2025 因子 IC 反转
- 详见 `memory/window8_collapse_analysis.md`

### 3. Exp 013 对比 (v4 单次训练)

| 配置 | Sharpe | MDD | RankIC |
|------|--------|-----|--------|
| alpha158_baseline | 1.264 | -26.9% | -0.002 |
| mutation_only | 1.067 | -27.5% | 0.029 |
| init_only | 1.158 | -26.4% | 0.031 |
| alpha158_mined | 1.098 | -26.4% | 0.029 |

### 4. Rolling SOTA (新排名 #1)

| 配置 | Sharpe | Return | MDD |
|------|--------|--------|-----|
| D_expand + alpha158_val_qa (278因子) | **2.180** | 106% | -12% |
| D_expand + alpha158_val (baseline) | 1.862 | 90% | -15% |

## 对框架的贡献

本论文复现推动了以下框架改进：

1. **`quanta/` 演化循环** — 5 步 Propose→Construct→Calculate→Backtest→Feedback
2. **Trace 系统** — HypothesisFeedback + DirectionTrace，记录演化轨迹
3. **因子池准入控制** — AST 去重 (`ast_dedup.py`) + 相关性筛选 (`factor_pool.py`)
4. **H5→bin 注入** — `inject_quantaalpha.py`，桥接 QuantaAlpha 输出与 Qlib 数据
5. **68 个新因子** — 形成 `alpha158_val_qa` preset (278因子)，Sharpe 2.180
6. **每日渐进式挖掘** — `--daily` 模式 + `DirectionRegistry` 跨天累积搜索

## 文件结构

```
papers/quantaalpha/
├── README.md                          # 本文件
├── 2602.07085v1.pdf                   # 原论文
├── scripts/
│   ├── run_paper_replication.py       # 论文复现主脚本
│   └── run_robustness_research.py     # Window 8 鲁棒性调研
└── results/
    ├── exp013_results.json            # Exp 013 v4 对比表
    ├── factor_pool_summary.json       # 14 因子入池详情
    └── rolling_sota.json              # Rolling SOTA (Sharpe 2.180)
```
