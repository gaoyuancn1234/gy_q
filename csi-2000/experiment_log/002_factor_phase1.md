# 实验 002: 因子扩展 Phase 1 — 量价因子增强

**日期**: 2026-02-08
**目标**: 在 Alpha158 基础上扩展量价因子，评估能否提升模型表现

## 实验设置

### 三个因子预设

| 预设 | 因子数 | 说明 |
|------|--------|------|
| alpha158 | 158 | 基线，Qlib 原始 Alpha158 |
| alpha158_ext | 208 | Alpha158 + 50个扩展量价因子 (全量) |
| alpha158_selected | 188 | Alpha158 + 30个精选扩展因子 (去冗余) |

### 新增 50 个扩展因子

| 分组 | 因子数 | 代表因子 |
|------|--------|---------|
| VWAP 系列 | 5 | VWAP, VWAP_RATIO, VWAP_MA5/10/20_RATIO |
| 换手率系列 | 8 | TURN_MA5/10/20/60, TURN_STD20, TURN_SURGE |
| 波动率增强 | 8 | ATR_14, GK_VOL_20, INTRADAY_RANGE, VOL_RATIO |
| 量价背离 | 6 | PV_CORR_10/20, VOLUME_SURPRISE, AMT_SURPRISE |
| 动量增强 | 8 | MOM_SKIP1_5/10/20, REVERSION_5/20, MOM_3M_1M |
| Alpha101 精选 | 15 | WQ_ALPHA6/12/15/22/26/28/33/38/41/45/54/68/73/84/101 |

数据来源: BaoStock 扩展字段 (amount/turn/pctChg) + Qlib 表达式引擎组合

## 步骤 1: 因子 IC 评价

对50个扩展因子计算截面 Spearman IC (2024-01 ~ 2026-02)。

**Top 10 因子:**

| 因子 | mean_IC | |ICIR| | 含义 |
|------|---------|-------|------|
| VWAP | -0.077 | 2.076 | 日内成交均价 |
| ATR_14 | -0.034 | 1.029 | 14日真实波动率 |
| TURN_MA60 | -0.032 | 0.822 | 60日均换手率 |
| MOM_SKIP1_5 | -0.034 | 0.791 | 跳1天的5日动量 |
| REVERSION_20 | 0.029 | 0.779 | 20日反转 |
| VWAP_MA20_RATIO | 0.030 | 0.760 | 20日VWAP均值/收盘价 |
| WQ_ALPHA22 | 0.026 | 0.577 | -corr(high, volume, 5) * std(close, 5) |
| TURN_MA10 | -0.023 | 0.549 | 10日均换手率 |
| MOM_SKIP1_10 | -0.017 | 0.471 | 跳1天的10日动量 |
| VOL_RATIO_5_20 | -0.019 | 0.461 | 短期/长期波动率比 |

信号质量: 13/50 因子 |ICIR| > 0.5, 仅 2/50 因子 |ICIR| > 1.0

## 步骤 2: 相关性分析与去冗余

### 相关性分布

- 均值: 0.174, 中位数: 0.112, 最大: 0.996
- corr > 0.7: 32 对, corr > 0.5: 99 对

### 高相关因子对 (Top 5)

| 因子A | 因子B | 相关性 |
|-------|-------|--------|
| TURN_SURGE | VOLUME_SURPRISE | 0.996 |
| WQ_ALPHA33 | WQ_ALPHA38 | 0.958 |
| WQ_ALPHA38 | WQ_ALPHA101 | 0.957 |
| MOM_SKIP1_20 | REVERSION_20 | 0.931 |
| TURN_RATIO_5_20 | TURN_RANK_5 | 0.928 |

### 去冗余结果 (阈值 0.7)

50 → **30 因子**, 移除 20 个冗余因子:
ATR_RATIO, CLOSE_RANGE_POS, GK_VOL_20, INTRADAY_RANGE, MOM_SKIP1_20, REVERSION_5, TURN_MA20, TURN_MA5, TURN_RANK_5, TURN_STD20, TURN_SURGE, UP_RATIO_20, UP_VOL_RATIO_20, VOLUME_SURPRISE, VWAP_MA10_RATIO, VWAP_MA5_RATIO, WQ_ALPHA101, WQ_ALPHA33, WQ_ALPHA45, WQ_ALPHA54

## 步骤 3: A/B 对比实验

### 完整结果表

| 预设 | 模型 | 总收益 | 年化 | Sharpe | 最大回撤 | 因子数 |
|------|------|--------|------|--------|---------|--------|
| alpha158 | LightGBM | 180.32% | 66.58% | 1.572 | -23.82% | 158 |
| alpha158 | XGBoost | 228.55% | 80.20% | 1.776 | -23.37% | 158 |
| alpha158_ext | LightGBM | 87.50% | 36.51% | 1.767 | -14.20% | 208 |
| alpha158_ext | XGBoost | 79.53% | 33.61% | 1.712 | -12.57% | 208 |
| alpha158_ext | CatBoost | 75.83% | 32.24% | 1.742 | -16.20% | 208 |
| alpha158_ext | Ridge | 140.88% | 54.54% | 1.696 | -19.52% | 208 |
| alpha158_ext | DoubleEnsemble | 65.87% | 28.47% | 1.376 | -18.85% | 208 |
| **alpha158_selected** | **LightGBM** | **124.05%** | **49.09%** | **2.157** | **-13.36%** | **188** |
| **alpha158_selected** | **CatBoost** | **94.00%** | **38.83%** | **2.111** | **-15.55%** | **188** |
| **alpha158_selected** | **XGBoost** | **98.15%** | **40.29%** | **1.981** | **-14.54%** | **188** |

### Sharpe 变化对比

```
LightGBM:  alpha158 (1.572) → ext (1.767) → selected (2.157)  ↑ 37%
XGBoost:   alpha158 (1.776) → ext (1.712) → selected (1.981)  ↑ 12%
CatBoost:  alpha158_ext (1.742) → selected (2.111)             ↑ 21%
```

### 最大回撤变化对比

```
LightGBM:  alpha158 (-23.82%) → selected (-13.36%)  改善 44%
XGBoost:   alpha158 (-23.37%) → selected (-14.54%)  改善 38%
CatBoost:  alpha158_ext (-16.20%) → selected (-15.55%)  改善 4%
```

## 结论

### 核心发现

1. **因子去冗余 > 因子堆量**: 50 因子全量加入 (alpha158_ext) 反而不如精选 30 因子 (alpha158_selected)。冗余因子引入噪声，干扰模型学习

2. **扩展因子的核心价值是风控**: alpha158 总收益更高 (228%)，但回撤也大 (-23%)。加入精选扩展因子后回撤降到 -13%，Sharpe 大幅提升。VWAP/ATR/换手率提供的是风险信号而非纯收益信号

3. **少即是多**: 158 → 208 (全量) 性能下降，158 + 30 (精选) = 188 达到最优

4. **最强新因子**: VWAP (ICIR=2.08) 和 ATR_14 (ICIR=1.03) 是最有价值的新增因子

5. **DoubleEnsemble 在高维因子上退化**: 208 因子时 Sharpe 仅 1.376，可能因内部特征选择与大量冗余因子冲突

### 最佳配置

**alpha158_selected + LightGBM**: Sharpe 2.157, 年化 49.09%, 最大回撤 -13.36%

### 遗留问题

- alpha158_selected 上尚未跑 DoubleEnsemble 和 DL 模型
- Phase 2 (基本面因子) 需要 Tushare Token

## 文件位置

- 因子定义: `factor_lab/factors/alpha158_ext.py`
- 预设配置: `factor_lab/factors/presets.py`
- IC 报告: `factor_lab/results/factor_eval/alpha158_ext_ic_report.csv`
- 相关性矩阵: `factor_lab/results/factor_eval/alpha158_ext_correlation.csv`
- 去冗余结果: `factor_lab/results/factor_eval/factor_selection_result.json`
- 回测结果: `factor_lab/results/experiments/*.json`
