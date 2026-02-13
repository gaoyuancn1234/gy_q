# 实盘化 + 策略增强 实施计划

## 总览

基于 Experiment 006 最优方案 (LGB topk=20 weekly SL=8%, Sharpe 2.054)，分 4 个 Phase 实施。

---

## Phase 1: 信号生成器 + 模拟盘引擎

### 1.1 信号生成器 `factor_lab/signal_generator.py`

核心职责：封装完整的 Rolling 预测 → TopK 信号 pipeline

```
类: SignalGenerator
├── __init__(config_path | config_dict)
├── load_or_train_model()          # 加载缓存模型 or 训练新模型
├── generate_signal(date=None)     # 生成当日/指定日信号
│   ├── 确定当前所处的 rolling window
│   ├── 如模型过期(>3个月) → 自动 retrain
│   ├── build handler + dataset (仅 infer segment)
│   ├── model.predict → rank → topK
│   └── 返回 {stock: score} + signal_quality metrics
├── check_signal_quality()         # IC 衰减监控
│   ├── 最近 20 天 IC 均值
│   ├── 与历史 IC 对比
│   └── 返回 quality_level: good/warning/degraded
└── retrain(force=False)           # 手动/自动 retrain
    ├── 用最新数据重建 expanding window
    ├── 训练 LightGBM
    ├── 保存模型 + 元数据
    └── 记录 retrain 日志
```

配置文件: `config/signal_config.yaml`
```yaml
model: LightGBM
preset: alpha158_val
rolling_config: D_expand_3v_3r
topk: 20
retrain_months: 3
model_cache_dir: factor_lab/models/
```

### 1.2 模拟盘引擎 `factor_lab/paper_trader.py`

核心职责：无真实交易的完整投资组合模拟

```
类: PaperTrader
├── __init__(config)
│   └── 加载持久化状态 or 初始化
├── State (持久化到 JSON):
│   ├── cash: float
│   ├── positions: {stock: {shares, cost_price, entry_date}}
│   ├── daily_nav: [{date, nav, benchmark_nav}]
│   ├── trade_history: [{date, stock, action, shares, price, reason}]
│   └── metadata: {start_date, initial_cash, strategy_params}
│
├── update_daily()                 # 每日收盘后调用
│   ├── 获取最新价格 (Qlib D.features)
│   ├── 更新持仓市值 + NAV
│   ├── 检查止损 → 生成止损卖出指令
│   ├── 如果是调仓日 → 调用 signal_generator → 生成交易指令
│   ├── 执行交易指令 (模拟: T+1 开盘价, 涨跌停检查)
│   └── 保存状态
│
├── rebalance(signal)              # 执行调仓
│   ├── 对比当前持仓 vs 目标持仓
│   ├── 生成卖出/买入指令
│   ├── 模拟执行 (次日开盘价)
│   └── 记录交易明细
│
├── get_performance()              # 绩效报告
│   ├── 总收益/年化/Sharpe/MDD
│   ├── 超额收益 vs 沪深300
│   ├── 月度收益分布
│   └── 当前持仓明细
│
├── get_trade_instructions()       # 生成人类可读的调仓指令
│   └── 格式: "卖出 XX 股 YY, 买入 XX 股 ZZ"
│
└── export_report(format='md')     # 导出报告
    ├── Markdown 格式
    ├── 含净值曲线描述
    └── 每周推送到飞书
```

状态持久化: `factor_lab/paper_trading/state.json`
交易记录: `factor_lab/paper_trading/trades.csv`
日净值: `factor_lab/paper_trading/daily_nav.csv`

---

## Phase 2: 调度与集成

### 2.1 定时调度器 `factor_lab/scheduler.py`

使用 APScheduler 实现 (或退化为 cron wrapper)：

```
每个交易日 15:30:
  1. 更新 Qlib 数据 (BaoStock 增量下载)
  2. paper_trader.update_daily()
  3. 如遇止损 → 飞书通知

每周五 16:00:
  1. signal_generator.generate_signal()
  2. paper_trader.rebalance(signal)
  3. 生成调仓指令 → 飞书通知
  4. 更新周报

每季度首月 1 日:
  1. signal_generator.retrain()
  2. 信号质量对比 (新模型 vs 旧模型)
  3. 飞书通知 retrain 结果
```

### 2.2 smart_bot 集成

在 `smart_bot.py` 的 `handle_quick_commands()` 中增加：

```
"模拟盘" / "纸盘" → 显示模拟盘绩效
"模拟信号"        → 手动触发信号生成
"模拟调仓"        → 手动触发调仓
"模拟报告"        → 导出完整报告
```

### 2.3 trade_executor.py 更新

新增 `STRATEGY_TYPE = 'rolling_qlib'`：
- 调用 SignalGenerator 而非旧的 QlibEngine
- 支持周度调仓 + 止损
- 输出格式兼容现有飞书卡片

---

## Phase 3: 策略增强

### 3.1 行业中性化 `factor_lab/strategy/industry_neutral.py`

```python
def neutralize_signal(signal, industry_map):
    """行业内排名，避免行业集中"""
    # 1. 按行业分组
    # 2. 行业内 zscore
    # 3. 每个行业选 topK/n_industries 只
    # 4. 确保不超过 30% 单行业暴露
```

需要: 行业分类数据 (申万一级, 可从 BaoStock 获取)

### 3.2 动态止损 `factor_lab/strategy/dynamic_stoploss.py`

```python
def atr_stop_loss(positions, prices, atr_multiplier=2.0):
    """基于 ATR 的自适应止损"""
    # 高波动股: 止损更宽 (避免 whipsaw)
    # 低波动股: 止损更紧 (及时止损)
    # ATR = 14日真实波幅均值
    # 止损线 = entry_price - atr_multiplier * ATR
```

### 3.3 信号强度加权 `factor_lab/strategy/signal_weighting.py`

```python
def signal_weighted_allocation(signal, topk=20):
    """按信号强度分配仓位权重"""
    # 等权: 每只 5%
    # 信号加权: top1 可能 8%, top20 可能 3%
    # 约束: 单只不超过 POSITION_LIMIT (15%)
```

---

## Phase 4: 因子发掘代理

### 4.1 因子发掘器 `factor_lab/factor_discovery.py`

```
类: FactorDiscoveryAgent
├── search_papers(keywords, source='arxiv+ssrn')
│   ├── 搜索最新量化论文/博客
│   ├── 提取因子公式和逻辑
│   └── 返回候选因子列表
│
├── parse_factor(paper_text)
│   ├── 用 LLM 提取因子定义
│   ├── 转化为 Qlib 表达式
│   └── 验证表达式语法
│
├── evaluate_candidates(candidates)
│   ├── 计算 IC/ICIR
│   ├── 与现有因子去相关
│   ├── 过滤冗余因子
│   └── 返回有效新因子
│
├── run_discovery_cycle()          # 完整发掘周期
│   ├── 搜索 → 解析 → 评估 → 报告
│   └── 保存发现日志
│
└── register_new_factors(approved)
    ├── 加入 FactorRegistry
    ├── 更新 preset
    └── 触发 retrain
```

### 4.2 定期运行

```
每月 1 日:
  1. factor_discovery.run_discovery_cycle()
  2. 生成因子发掘报告 → 飞书通知
  3. 如有高质量新因子 → 人工确认后加入
```

---

## 文件结构变更

```
trading_framework/
├── factor_lab/
│   ├── signal_generator.py      [新建] Phase 1
│   ├── paper_trader.py          [新建] Phase 1
│   ├── scheduler.py             [新建] Phase 2
│   ├── factor_discovery.py      [新建] Phase 4
│   ├── strategy/                [新建目录] Phase 3
│   │   ├── __init__.py
│   │   ├── industry_neutral.py
│   │   ├── dynamic_stoploss.py
│   │   └── signal_weighting.py
│   ├── paper_trading/           [新建目录] Phase 1
│   │   ├── state.json           (运行时生成)
│   │   ├── trades.csv
│   │   └── daily_nav.csv
│   └── models/                  [新建目录] Phase 1
│       └── (trained model cache)
├── config/
│   ├── settings.py              [修改] Phase 2
│   └── signal_config.yaml       [新建] Phase 1
├── portfolio/
│   └── trade_executor.py        [修改] Phase 2
└── smart_bot.py                 [修改] Phase 2
```

---

## 实施顺序

1. **Phase 1** — signal_generator.py + paper_trader.py (核心)
2. **Phase 2** — scheduler.py + smart_bot 集成 + trade_executor 更新
3. **Phase 3** — 行业中性 + 动态止损 + 信号加权 (可并行)
4. **Phase 4** — 因子发掘代理

先做 Phase 1，跑通核心链路后再逐步扩展。
