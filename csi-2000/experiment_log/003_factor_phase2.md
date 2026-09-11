# 实验 003: 因子扩展 Phase 2 — 基本面因子

**日期**: 2026-02-08
**目标**: 在 alpha158_selected (量价去冗余) 基础上加入 PE/PB/市值等基本面因子，评估能否进一步提升

## 数据准备

### 数据源: Tushare daily_basic

通过 Tushare Pro API 按交易日批量下载全市场基本面数据。

| 字段 | 含义 | 覆盖率 |
|------|------|--------|
| pe_ttm | 滚动市盈率 | 98.3% |
| pb | 市净率 | 99.7% |
| ps_ttm | 滚动市销率 | 98.3% |
| total_mv | 总市值 (万元→元) | 99.7% |
| circ_mv | 流通市值 (万元→元) | 99.7% |

- **数据范围**: 2018-01-02 ~ 2026-02-07 (1967 交易日)
- **股票数**: 300 (CSI300)
- **总行数**: 540,916
- **整体完整度**: 99.3%

注入方式: 写入 `~/.qlib/qlib_data/cn_data_bs/features/{stock}/{field}.day.bin`，注入后可在 Qlib 表达式中用 `$pe_ttm` 等引用。

## 因子设计: 22 个基本面因子

### 估值因子 (17个)

| 因子 | 表达式 | 含义 |
|------|--------|------|
| PE_TTM | `$pe_ttm` | 滚动市盈率 |
| PB | `$pb` | 市净率 |
| PS_TTM | `$ps_ttm` | 滚动市销率 |
| EP | `1/($pe_ttm+1e-8)` | 盈利收益率 (1/PE) |
| BP | `1/($pb+1e-8)` | 账面价值率 (1/PB) |
| SP | `1/($ps_ttm+1e-8)` | 销售收益率 (1/PS) |
| LOG_MV | `Log($total_mv+1)` | 对数总市值 |
| LOG_CIRC_MV | `Log($circ_mv+1)` | 对数流通市值 |
| MV_RATIO | `$circ_mv/($total_mv+1e-8)` | 流通/总市值比 |
| PE_MA20 | `Mean($pe_ttm, 20)` | 20日PE均值 |
| PE_STD20 | `Std($pe_ttm, 20)` | 20日PE标准差 |
| PE_ZSCORE | 60日Z-score | PE 估值分位 |
| PB_ZSCORE | 60日Z-score | PB 估值分位 |
| PE_CHANGE_20 | 20日变化率 | PE动态变化 |
| PB_CHANGE_20 | 20日变化率 | PB动态变化 |
| PE_PRICE_CORR | `Corr($pe_ttm, $close, 20)` | PE与价格相关性 |

### 市值因子 (3个)

| 因子 | 表达式 | 含义 |
|------|--------|------|
| MV_RANK | `Rank($total_mv, 20)` | 市值滚动排名 |
| NEG_LOG_MV | `-1*Log($total_mv+1)` | 负对数市值 (小市值因子) |
| MV_CHANGE_20 | 20日变化率 | 市值变化速度 |

### 估值×量价交叉因子 (3个)

| 因子 | 表达式 | 含义 |
|------|--------|------|
| EP_MOM20 | EP × 20日动量 | 价值动量 |
| BP_TURN | BP × 换手率 | 估值+活跃度 |
| MV_VOL | Log(MV) × 20日波动率 | 市值波动交互 |

## 因子 IC 评价

评价期: 2024-01-02 ~ 2026-02-07, CSI300, 300 交易日

### Top 10 因子 (按 |ICIR| 排序)

| 因子 | mean_IC | |ICIR| | IC>0 比例 | 信号方向 |
|------|---------|-------|-----------|----------|
| NEG_LOG_MV | +0.076 | **2.116** | 98.7% | 小市值 → 高收益 |
| LOG_MV | -0.076 | 2.116 | 1.3% | (反向) |
| LOG_CIRC_MV | -0.073 | 1.991 | 1.7% | (反向) |
| BP | +0.077 | **1.979** | 98.3% | 高BP → 高收益 |
| PB | -0.077 | 1.979 | 1.7% | (反向) |
| SP | +0.071 | **1.846** | 95.3% | 高SP → 高收益 |
| PS_TTM | -0.071 | 1.846 | 4.7% | (反向) |
| EP | +0.057 | **1.197** | 89.2% | 高EP → 高收益 |
| PE_TTM | -0.057 | 1.197 | 10.8% | (反向) |
| PE_MA20 | -0.046 | 1.040 | 13.6% | (反向) |

### 信号质量评估

- |ICIR| > 1.0 的因子: **10/22** (45%)
- |ICIR| > 0.5 的因子: **16/22** (73%)
- IC 方向一致性 > 90%: **10/22**

**对比 Phase 1 量价因子**: Phase 1 仅 2/50 因子 |ICIR| > 1.0，Phase 2 有 10/22 因子 |ICIR| > 1.0。基本面因子的预测信号远强于量价扩展因子。

### 关键发现

1. **小市值效应** (ICIR=2.12) 是 A 股最强的横截面因子
2. **价值因子** (BP/SP/EP) 均有效，BP 最强 (ICIR=1.98)
3. 原始估值 (PE/PB/PS) 和倒数形式 (EP/BP/SP) 互为镜像，模型只需一种
4. 交叉因子 (EP_MOM20/BP_TURN/MV_VOL) 信号较弱 (ICIR < 0.8)

## A/B 对比实验

### 预设对比

| 预设 | 因子数 | 组成 |
|------|--------|------|
| alpha158_selected | 188 | Alpha158 + 30 精选量价 |
| alpha158_val | 210 | Alpha158 + 30 精选量价 + 22 基本面 |

### 完整结果表

| 预设 | 模型 | 总收益 | 年化 | Sharpe | 最大回撤 | 超额收益 |
|------|------|--------|------|--------|---------|----------|
| alpha158_selected | LightGBM | 124.05% | 49.09% | 2.157 | -13.36% | 86.44% |
| alpha158_selected | XGBoost | 98.15% | 40.29% | 1.981 | -14.54% | 60.54% |
| alpha158_selected | CatBoost | 94.00% | 38.83% | 2.111 | -15.55% | 56.39% |
| **alpha158_val** | **LightGBM** | **132.76%** | **51.93%** | **2.192** | **-15.13%** | **95.15%** |
| alpha158_val | XGBoost | 94.20% | 38.90% | 1.791 | -13.97% | 56.60% |
| alpha158_val | CatBoost | 97.42% | 40.04% | 1.936 | -13.04% | 59.81% |

### Sharpe 变化

```
LightGBM:  selected (2.157) → val (2.192)  ↑ +0.035 (+1.6%)
XGBoost:   selected (1.981) → val (1.791)  ↓ -0.190 (-9.6%)
CatBoost:  selected (2.111) → val (1.936)  ↓ -0.175 (-8.3%)
```

### 最大回撤变化

```
LightGBM:  selected (-13.36%) → val (-15.13%)  恶化 +1.77%
XGBoost:   selected (-14.54%) → val (-13.97%)  改善 -0.57%
CatBoost:  selected (-15.55%) → val (-13.04%)  改善 -2.51%
```

### 总收益变化

```
LightGBM:  selected (124.05%) → val (132.76%)  ↑ +8.71%
XGBoost:   selected (98.15%)  → val (94.20%)   ↓ -3.95%
CatBoost:  selected (94.00%)  → val (97.42%)   ↑ +3.42%
```

## 结论

### 核心发现

1. **基本面因子信号极强但模型表现分化**:
   - 基本面因子单独评价时信号远强于量价因子 (10/22 ICIR>1.0 vs 2/50)
   - 但加入模型后效果因模型而异: LightGBM 受益，XGBoost/CatBoost Sharpe 下降

2. **LightGBM 继续领先**: alpha158_val + LightGBM 达到全系列最高 Sharpe (2.192)，总收益 132.76%，是目前最佳配置

3. **风控改善**: CatBoost 回撤从 -15.55% 改善到 **-13.04%** (全系列最低)，XGBoost 也小幅改善。基本面因子为模型提供了估值锚定

4. **维度诅咒初现**: 因子从 188 增到 210 (+22 基本面)，XGBoost/CatBoost 的 Sharpe 反而下降。这与 Phase 1 从 208→188 (去冗余提升) 的规律一致 — 树模型在因子数增加时容易过拟合

5. **基本面因子内部存在冗余**: PE/PB/PS 与其倒数 EP/BP/SP 完全互为镜像 (corr=-1.0)，LOG_MV 与 LOG_CIRC_MV 高度相关。22 个因子中有效独立信号可能只有 8-10 个

### 建议的最佳配置

| 用途 | 推荐配置 | Sharpe | 最大回撤 |
|------|----------|--------|---------|
| 追求收益 | alpha158_val + LightGBM | 2.192 | -15.13% |
| 追求风控 | alpha158_val + CatBoost | 1.936 | -13.04% |
| 均衡首选 | alpha158_selected + LightGBM | 2.157 | -13.36% |

### 下一步 (Phase 3)

- 对 22 个基本面因子做去冗余，减少到 ~10 个独立因子
- 引入资金流/北向/融资融券数据 (full preset)
- 在 alpha158_val 上跑 DoubleEnsemble 和 DL 模型

## 文件位置

- 数据下载: `factor_lab/data/download_tushare_fundamental.py`
- 因子定义: `factor_lab/factors/fundamental.py`
- 预设配置: `factor_lab/factors/presets.py`
- IC 报告: `factor_lab/results/factor_eval/fundamental_ic_report.csv`
- 回测结果: `factor_lab/results/experiments/alpha158_val_*.json`
- 数据缓存: `factor_lab/results/.cache/tushare_daily_basic.parquet`
