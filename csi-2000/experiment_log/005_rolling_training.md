# 实验 005: Rolling Training 滚动训练对比

## 实验目的

评估滚动训练 (Walk-Forward) 相比单次训练 (Single-Shot) 的增益：
1. **适应市场变化**: 模型定期用最新数据重训，捕捉市场结构变化
2. **利用新数据源**: 后期窗口可学习资金流因子 (仅 2025-08 后有数据)
3. **减少过拟合**: 多个窗口的预测拼接，降低单次训练的偶然性

## 实验设计

### Baseline (Single-Shot)
- 训练: 2019-01-01 ~ 2023-06-30
- 验证: 2023-07-01 ~ 2023-12-31
- 测试: 2024-01-01 ~ 2026-02-05
- 最佳: alpha158_val + LightGBM → Sharpe 2.192

### Rolling 配置矩阵

| 配置 | 训练窗口 | 验证期 | 重训频率 | 预计窗口数 |
|------|---------|--------|---------|-----------|
| A_4yr_3v_3r | 4年滑动 | 3个月 | 3个月 | ~9 |
| B_4yr_3v_6r | 4年滑动 | 3个月 | 6个月 | ~5 |
| C_3yr_3v_3r | 3年滑动 | 3个月 | 3个月 | ~9 |
| D_expand_3v_3r | 扩展窗口 | 3个月 | 3个月 | ~9 |

### 模型
LightGBM, XGBoost, CatBoost (树模型, 超参与 benchmark_models.py 一致)

### 因子预设
- alpha158_val (主要): Alpha158 + 去冗余量价 + 基本面 (~210因子)
- alpha158 (对照): 原始 Alpha158 (158因子)
- full (验证): 含资金流因子 (~231因子)

## 实验步骤

```bash
# Step 1: 快速验证
python -m factor_lab.run_rolling_benchmark --configs A_4yr_3v_3r --presets alpha158_val --models LightGBM

# Step 2: 核心对比 (4 configs × LightGBM)
python -m factor_lab.run_rolling_benchmark --presets alpha158_val --models LightGBM

# Step 3: 扩展模型 (最佳 config × 3 models)
python -m factor_lab.run_rolling_benchmark --configs <best> --models LightGBM XGBoost CatBoost

# Step 4: 全量 (选择性)
python -m factor_lab.run_rolling_benchmark
```

## 结果

### Step 1: 快速验证

**Config A_4yr_3v_3r × alpha158_val × LightGBM**

| 指标 | Rolling (9窗口) | Single-Shot |
|------|----------------|-------------|
| Sharpe | **2.093** | 1.572 |
| 总收益 | 108.01% | 180.32% |
| 年化 | 43.71% | 66.58% |
| 超额 | 70.41% | 142.71% |
| 最大回撤 | **-10.97%** | -23.82% |
| 耗时 | 456s | 29s |

观察:
- Rolling 显著提升 Sharpe (+33%) 和回撤控制 (-54%)
- 但总收益下降 (-40%)，因为模型更保守
- Window 8 (2025Q4) best_iter=1 说明信号极弱 (验证集 loss 上升)
- Window 2-3 best_iter=469/499 说明信号强，模型能持续学习

### Step 2: 核心对比 (4 configs × alpha158_val × LightGBM)

| 排名 | Config | Sharpe | 总收益 | 年化 | 超额 | 最大回撤 | 耗时 |
|------|--------|--------|--------|------|------|---------|------|
| 1 | **D_expand_3v_3r** | **2.202** | 117.48% | 46.91% | 79.87% | -12.55% | 507s |
| 2 | B_4yr_3v_6r | 2.096 | 111.24% | 44.81% | 73.64% | -11.22% | 285s |
| 3 | A_4yr_3v_3r | 2.093 | 108.01% | 43.71% | 70.41% | -10.97% | 456s |
| 4 | C_3yr_3v_3r | 2.020 | 106.06% | 43.04% | 68.45% | -11.88% | 391s |
| - | single-shot | 1.572 | 180.32% | 66.58% | 142.71% | -23.82% | 29s |

观察:
- **所有 rolling 配置的 Sharpe 都优于 single-shot** (2.0~2.2 vs 1.57)
- 扩展窗口 (D) 最优: 训练数据越多模型越稳定
- 4年训练窗口 (A) > 3年 (C): 更长历史有助于信号提取
- 6月重训 (B) vs 3月重训 (A): Sharpe 相近，但 B 更快 (5窗口 vs 9窗口)
- 所有 rolling 的回撤控制都远优于 single-shot (-11%~-13% vs -24%)

### Step 3: 模型对比 (D_expand × 3 models × alpha158_val)

| 排名 | Model | Sharpe | 总收益 | 年化 | 超额 | 最大回撤 | 耗时 |
|------|-------|--------|--------|------|------|---------|------|
| 1 | **LightGBM** | **2.202** | 117.48% | 46.91% | 79.87% | -12.55% | 507s |
| 2 | CatBoost | 2.038 | 91.37% | 37.89% | 53.76% | **-10.24%** | 593s |
| 3 | XGBoost | 1.701 | 75.70% | 32.18% | 38.09% | -12.54% | 571s |

观察:
- LightGBM 在 rolling 训练中仍然是最佳模型
- CatBoost 回撤最小 (-10.24%)，风格最保守
- XGBoost rolling 表现最弱，但仍优于 single-shot baseline
- Window 8 (2025Q4): 所有模型 best_iter 极低 (1~3)，该时段市场信号极弱

## 结论

### 核心发现

1. **Rolling 训练显著改善风险调整收益**: 所有配置的 Sharpe 都从 1.57 提升到 2.0+
2. **回撤控制大幅改善**: 最大回撤从 -24% 降至 -10%~-13%，降低 46%~57%
3. **总收益有所牺牲**: Rolling 总收益 75%~117% vs single-shot 180%，原因是模型更保守、部分窗口信号弱
4. **最优配置**: D_expand_3v_3r + LightGBM + alpha158_val → **Sharpe 2.202**, MDD -12.55%
5. **扩展窗口优于滑动窗口**: 训练数据越多，模型越稳定
6. **2025Q4 信号衰减**: Window 8 所有模型 early-stop 极早 (1~3轮)，市场结构可能发生变化

### 实际应用建议

- 生产环境推荐使用 **D_expand_3v_3r** (扩展窗口，每3月重训)
- 若需更快回测速度，**B_4yr_3v_6r** (半年重训) 是合理替代
- Rolling 训练的主要价值是**降低回撤风险**，而非提升绝对收益
