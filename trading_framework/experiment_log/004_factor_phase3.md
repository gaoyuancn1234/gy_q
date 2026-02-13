# 实验 004: 因子扩展 Phase 3 — 资金流 / 北向资金 / 融资融券

**日期**: 2026-02-08
**目标**: 在 alpha158_val (量价精选 + 基本面) 基础上加入资金流/北向/融资融券数据，构建完整因子体系

## 背景

Phase 1-2 结论:
- **最佳配置**: alpha158_val + LightGBM: Sharpe 2.192, MDD -15.13%, 总收益 132.76%
- **最低回撤**: alpha158_val + CatBoost: Sharpe 1.936, MDD -13.04%
- 基本面因子信号极强 (10/22 ICIR>1.0)
- 维度诅咒: 因子数增加时部分模型 Sharpe 下降

## 数据准备

### 数据源 1: Tushare 北向资金 (市场级)

| 字段 | 含义 | 类型 |
|------|------|------|
| north_money | 北向资金总额 (沪股通+深股通) | 市场级 (所有股票共享) |

- **下载方式**: `moneyflow_hsgt()` 一次返回全部日期
- **数据范围**: 2018-01-01 ~ 2026-02-08
- **注入方式**: `inject_market_field()` → 每只股票复制同一值
- **脚本**: `factor_lab/data/download_north_money.py`

### 数据源 2: AKShare 个股资金流向

| 字段 | 含义 | 覆盖率 |
|------|------|--------|
| main_net_inflow | 主力净流入 | ~90% |
| super_large_net | 超大单净流入 | ~90% |
| large_net | 大单净流入 | ~90% |
| medium_net | 中单净流入 | ~90% |
| small_net | 小单净流入 | ~90% |

- **下载方式**: 逐只下载, 80只/批, 批间120s
- **预计耗时**: ~5小时
- **脚本**: `factor_lab/data/download_akshare_fund_flow.py`

### 数据源 3: AKShare 融资融券

| 字段 | 含义 | 覆盖率 |
|------|------|--------|
| margin_balance | 融资余额 | ~70-80% |
| short_balance | 融券余额 | ~70-80% |

- **说明**: 非所有股票都是融资融券标的
- **预计耗时**: ~4小时
- **脚本**: `factor_lab/data/download_akshare_margin.py`

## 因子设计: 24 个资金流因子 (5组)

### 主力资金流向 (6个)

| 因子 | 表达式 | 含义 |
|------|--------|------|
| MAIN_NET_5 | `Mean($main_net_inflow, 5)` | 5日主力净流入均值 |
| MAIN_NET_10 | `Mean($main_net_inflow, 10)` | 10日主力净流入均值 |
| MAIN_NET_20 | `Mean($main_net_inflow, 20)` | 20日主力净流入均值 |
| MAIN_NET_ACC5 | `Sum($main_net_inflow, 5)` | 5日主力净流入累计 |
| MAIN_NET_ACC20 | `Sum($main_net_inflow, 20)` | 20日主力净流入累计 |
| MAIN_NET_CHANGE | `Mean(5) - Mean(20)` | 主力净流入短-长差异 |

### 大单/小单分析 (4个)

| 因子 | 含义 |
|------|------|
| BIG_ORDER_RATIO | 大单占比 |
| SUPER_LARGE_5 | 5日超大单净流入 |
| SMALL_NET_5 | 5日小单净流入 (散户行为) |
| BIG_SMALL_RATIO | 大单/小单净流入比 |

### 北向资金 (5个)

| 因子 | 含义 |
|------|------|
| NORTH_MA5 | 5日北向资金均值 |
| NORTH_MA20 | 20日北向资金均值 |
| NORTH_ACC5 | 5日北向资金累计 |
| NORTH_CHANGE | 北向资金短-长差异 |
| NORTH_MOM | 北向资金5日动量 |

### 融资融券 (6个)

| 因子 | 含义 |
|------|------|
| MARGIN_BAL | 融资余额 |
| MARGIN_CHANGE_5 | 融资余额5日变化率 |
| MARGIN_CHANGE_20 | 融资余额20日变化率 |
| SHORT_BAL | 融券余额 |
| MARGIN_SHORT_RATIO | 融资/融券余额比 |
| NET_MARGIN | 融资融券净额 |

### 交叉因子 (3个)

| 因子 | 含义 |
|------|------|
| MAIN_PRICE_CORR | 10日主力资金-价格相关性 |
| MAIN_VOL_CORR | 10日主力资金-成交量相关性 |
| MARGIN_PRICE_CORR | 20日融资余额-价格相关性 |

## 预设配置

### 修改后的预设

| 预设 | 因子数 | 组成 |
|------|--------|------|
| alpha158 | 158 | 原始 Alpha158 |
| alpha158_selected | ~188 | Alpha158 + 30 精选量价 |
| alpha158_val | ~210 | Alpha158 + 30 精选量价 + 22 基本面 |
| **full** | **~231** | Alpha158 + 30 精选量价 + 22 基本面 + 21 资金流 (排除3个交叉因子) |
| **full_selected** | **TBD** | full 去冗余后最优子集 |

**关键修改**:
- `full` preset 改用精选量价 (30个) 而非全量 (50个)，避免维度诅咒
- 排除 3 个交叉因子 (Corr 操作符无法处理不同长度的 bin 文件)

## 数据覆盖情况

**重要限制**: AKShare 免费 API 仅返回最近 ~120 个交易日的数据。

| 数据源 | 覆盖范围 | 交易日数 |
|--------|----------|---------|
| Tushare 北向资金 | 2024-11-06 ~ 2026-02-06 | 300 |
| AKShare 个股资金流 | 2025-08-13 ~ 2026-02-06 | 120 |
| AKShare 融资融券 | 2025-08-13 ~ 2026-02-06 | 120 |

**影响**: 训练期 (2019-01-01 ~ 2023-06-30) 的资金流因子全为 NaN，模型无法学习这些因子的信号。Benchmark 中 full 与 alpha158_val 结果完全一致。

## IC 评价

评价期: 2025-08-13 ~ 2026-02-05, CSI300, 300只股票

### 北向资金因子 IC (5个)

| 因子 | mean_IC | ICIR | IC>0比例 | 信号方向 |
|------|---------|------|---------|----------|
| NORTH_MA20 | -0.034 | **-0.675** | 28.3% | 北向流入多 → 未来低收益 (逆向) |
| NORTH_MA5 | -0.027 | -0.532 | 30.0% | 同上 |
| NORTH_ACC5 | -0.024 | -0.478 | 31.0% | 同上 |
| NORTH_MOM | -0.004 | -0.080 | 49.3% | 极弱 |
| NORTH_CHANGE | +0.003 | +0.072 | 52.7% | 极弱 |

### 融资融券因子 IC (6个)

| 因子 | mean_IC | ICIR | IC>0比例 |
|------|---------|------|---------|
| SHORT_BAL | -0.036 | **-0.401** | 34.7% |
| MARGIN_SHORT_RATIO | +0.025 | +0.290 | 59.7% |
| MARGIN_CHANGE_5 | -0.022 | -0.245 | 39.3% |
| MARGIN_BAL | -0.016 | -0.150 | 45.3% |
| NET_MARGIN | -0.016 | -0.147 | 45.0% |
| MARGIN_CHANGE_20 | -0.008 | -0.081 | 44.7% |

### 主力资金流/大单因子 IC (10个)

| 因子 | mean_IC | ICIR | IC>0比例 |
|------|---------|------|---------|
| BIG_ORDER_RATIO | -0.024 | -0.248 | 41.3% |
| BIG_SMALL_RATIO | -0.014 | -0.166 | 45.0% |
| MAIN_NET_10 | -0.013 | -0.162 | 43.7% |
| MAIN_NET_20 | -0.009 | -0.104 | 45.7% |
| MAIN_NET_CHANGE | +0.006 | +0.075 | 55.7% |
| SUPER_LARGE_5 | +0.005 | +0.067 | 50.3% |
| MAIN_NET_5 | +0.005 | +0.064 | 54.3% |
| SMALL_NET_5 | +0.001 | +0.012 | 48.7% |
| MAIN_NET_ACC20 | -0.001 | -0.017 | 50.0% |
| MAIN_NET_ACC5 | +0.006 | +0.078 | 55.3% |

### 分组汇总

| 分组 | 因子数 | 平均|IC| | 平均|ICIR| |
|------|--------|----------|-----------|
| 融资融券 | 6 | 0.078 | 0.219 |
| 北向资金 | 5 | 0.061 | 0.282 |
| 大单分析 | 4 | 0.070 | 0.124 |
| 主力资金 | 6 | 0.065 | 0.083 |

### 与 Phase 1/2 因子对比

| 阶段 | 因子类型 | 最强 ICIR | |ICIR|>1.0 占比 |
|------|----------|-----------|---------------|
| Phase 1 | 量价扩展 | 2.08 (VWAP) | 2/50 (4%) |
| Phase 2 | 基本面 | 2.12 (NEG_LOG_MV) | 10/22 (45%) |
| **Phase 3** | **资金流** | **0.675 (NORTH_MA20)** | **0/21 (0%)** |

**结论**: 资金流因子信号最弱，所有因子 |ICIR| < 1.0。最强的北向资金 MA20 仅 0.675。

## A/B 对比实验

### 预设对比

| 预设 | 模型 | 因子数 | Sharpe | 最大回撤 | 总收益 | 超额收益 |
|------|------|--------|--------|---------|--------|----------|
| alpha158_selected | LightGBM | 188 | 2.157 | -13.36% | 124.05% | 86.44% |
| alpha158_selected | CatBoost | 188 | 2.111 | -15.55% | 94.00% | 56.39% |
| alpha158_val | LightGBM | 210 | 2.192 | -15.13% | 132.76% | 95.15% |
| alpha158_val | CatBoost | 210 | 1.936 | -13.04% | 97.42% | 59.81% |
| **full** | **LightGBM** | **231** | **2.192** | **-15.13%** | **132.76%** | **95.15%** |
| **full** | **CatBoost** | **231** | **1.936** | **-13.04%** | **97.42%** | **59.81%** |

**full 与 alpha158_val 完全一致** — 因为资金流因子在训练期全为 NaN，模型完全忽略了这些因子。

## 结论

### 核心发现

1. **资金流因子信号最弱**: 21 个因子中无一 |ICIR| > 1.0，远弱于基本面因子 (45% 超过 1.0) 和量价因子 (4%)

2. **数据覆盖不足是主要瓶颈**: AKShare 免费 API 仅返回 120 天历史，完全无法覆盖训练期。需要付费数据源 (如 Wind/Choice) 获取完整历史

3. **北向资金是反向指标**: NORTH_MA20 (ICIR=-0.675) 表明北向大量流入后股价倾向下跌，可能反映了"聪明资金"的获利了结

4. **融资融券有微弱信号**: SHORT_BAL (ICIR=-0.401) 和 MARGIN_SHORT_RATIO (ICIR=+0.290) 有一定预测力，但强度有限

5. **Benchmark 无提升**: full preset (231 因子) 与 alpha158_val (210 因子) 结果完全相同，因为新增因子在训练期无数据

6. **交叉因子不可用**: Qlib 的 Corr 操作符无法处理不同长度的 bin 文件，已排除 3 个交叉因子

### 最佳配置 (维持 Phase 2 结论)

| 用途 | 推荐配置 | Sharpe | 最大回撤 |
|------|----------|--------|---------|
| 追求收益 | alpha158_val + LightGBM | 2.192 | -15.13% |
| 追求风控 | alpha158_val + CatBoost | 1.936 | -13.04% |
| 均衡首选 | alpha158_selected + LightGBM | 2.157 | -13.36% |

### 后续建议

1. **获取完整历史数据**: 使用 Wind/Choice 等付费数据源获取 2018 年以来的资金流/融资融券数据，重跑实验
2. **Tushare 积分升级**: 升级到更高积分获取完整北向资金历史 (当前仅 300 天)
3. **尝试仅在测试期使用资金流**: 通过 rolling training 让最近窗口包含资金流数据
4. **因子去冗余仍有价值**: 即使信号弱，去冗余可避免噪声因子干扰模型

## 执行步骤

1. [x] 创建下载脚本 (download_north_money.py, download_akshare_fund_flow.py, download_akshare_margin.py)
2. [x] 运行 Step 1: 北向资金下载+注入 — 300条, 覆盖 2024-11-06~2026-02-06
3. [x] 运行 Step 2: AKShare 资金流下载 — 300只, 120天, 6.7分钟
4. [x] 运行 Step 3: AKShare 融资融券下载 — 300只, 120天, 2.6分钟
5. [x] 运行 Step 4: 因子 IC 评价 — 21个因子, 最强 ICIR=0.675
6. [x] 运行 Step 5: full benchmark — full=alpha158_val (因子训练期无数据)
7. [ ] 运行 Step 6: 全因子去冗余 (需完整历史数据后才有意义)
8. [x] 更新本实验记录

## 文件位置

| 文件 | 说明 |
|------|------|
| `factor_lab/data/download_north_money.py` | 北向资金下载+注入 |
| `factor_lab/data/download_akshare_fund_flow.py` | 资金流下载+注入 |
| `factor_lab/data/download_akshare_margin.py` | 融资融券下载+注入 |
| `factor_lab/factors/money_flow.py` | 24个资金流因子定义 |
| `factor_lab/factors/presets.py` | 预设配置 (full/full_selected) |
| `factor_lab/run_money_flow_eval.py` | 因子 IC 评价 |
| `factor_lab/run_full_benchmark.py` | full benchmark |
| `factor_lab/run_full_dedup.py` | 全因子去冗余 |
| `factor_lab/results/factor_eval/money_flow_*_ic_report.csv` | IC 报告 |
| `factor_lab/results/experiments/full_*.json` | 回测结果 |
