# 实验 009: 实盘化 Phase 1 — SignalGenerator + PaperTrader

## 目标

基于实验 005-008 的最佳策略，构建可日常运行的信号生成器和模拟盘引擎。

**最佳策略**: D_expand_3v_3r + LightGBM + alpha158_val + 策略A TopK自适应

## 设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 成交价 | **收盘价** (T日信号 → T+1收盘价) | 更保守更真实，避免开盘跳空影响 |
| 信号源 | 复用已有 pkl 缓存 | 不重新训练，直接使用实验005的rolling预测 |
| 信号质量 | 复用实验008 quality_score.pkl | 策略A TopK自适应: strong→12, normal→16, weak→20 |
| 初始资金 | 100万 | 实盘级别资金规模 |

## 新建文件

| 文件 | 说明 |
|------|------|
| `config/signal_config.yaml` | 信号配置 (模型/交易参数/路径) |
| `factor_lab/signal_generator.py` | SignalGenerator 类 (信号生成 + 调仓指令) |
| `factor_lab/paper_trader.py` | PaperTrader 类 (模拟盘引擎 + replay) |
| `experiment_log/009_paper_trading.md` | 本文件 |

## 架构

```
signal_config.yaml
       │
       ▼
SignalGenerator ──→ load_predictions() (pkl缓存)
       │            load_quality_score() (pkl缓存)
       │            get_signal(date) → TopK信号
       │            get_rebalance_instructions()
       │
       ▼
PaperTrader ──→ update_daily() (逐日更新)
       │         replay() (批量回放验证)
       │         get_performance() (绩效计算)
       │
       ▼
paper_trading/
  ├── state.json      (持仓状态)
  ├── trades.csv      (交易记录)
  ├── daily_nav.csv   (日净值)
  └── replay_performance.json
```

## 执行模型 (收盘价 T+1)

```
Day T 收盘:
  1. 获取信号 → 选出 TopK 目标股
  2. 检查止损 (持仓跌幅 > 8%)
  3. 生成调仓指令 → pending_orders

Day T+1 收盘:
  4. 以 T+1 收盘价执行 pending_orders
  5. 涨跌停检查 (跌停不卖: ret < -9.5%, 涨停不买: ret > 9.5%)
  6. 更新持仓市值 + NAV
```

## Replay 验证结果

回放区间: 2024-01-02 ~ 2026-02-05 (509个交易日)

### 对比表

| 指标 | AdaptiveBacktester (开盘价, 1亿) | PaperTrader (收盘价, 100万) | 差异 |
|------|------|------|------|
| Sharpe | 2.055 | **1.736** | -15.5% |
| 总收益 | 109.3% | **70.2%** | -39.1pp |
| 年化收益 | 44.6% | **30.2%** | |
| 最大回撤 | -12.59% | **-9.62%** | 更优 |
| 基准收益 | — | 36.9% | |
| 超额收益 | — | **33.3%** | |
| 交易次数 | 1788 | 1762 | -1.5% |
| 信号状态 | s=19 n=41 w=42 | s=19 n=41 w=42 | 完全一致 |

### 差异分析

**收益降低原因 (109% → 70%)**:
1. **成交价差异**: 收盘价 vs 开盘价。A股常有开盘跳空 → 开盘价买入可能更优
2. **资金规模**: 100万 vs 1亿。整手 (100股) 约束在小资金下影响更大，部分股票因单手成本过高无法买入
3. **四舍五入效应**: 小资金下等权分配后的整手计算损失更大

**MDD 更优 (-9.64% vs -12.59%)**:
- 收盘价执行天然具有"滞后性"，避免了开盘跳空带来的极端波动
- 小资金整手约束导致单只股票权重上限更低，分散化更好

### 结论

- Sharpe 降幅 15.5%，主要因为成交价差异 + 资金规模效应
- MDD 反而改善 -9.62% (更好的风控属性)
- 超额收益 33.3% (vs 基准 36.9%)，年化超额约 14%
- 信号状态分布完全一致，验证了 SignalGenerator 与 AdaptiveBacktester 的信号逻辑一致性

## 用法

```bash
# 回放历史 (验证)
cd trading_framework
python -m factor_lab.paper_trader replay

# 查看持仓状态
python -m factor_lab.paper_trader status

# 重置模拟盘
python -m factor_lab.paper_trader reset
```

```python
# 获取当日信号
from factor_lab.signal_generator import SignalGenerator
sg = SignalGenerator()
signal = sg.get_signal('2026-02-05')
print(signal['target_stocks'])

# 检查模型新鲜度
print(sg.check_model_freshness())
```

## 后续 (Phase 2)

- [ ] 接入飞书推送 (每日信号 + 调仓指令)
- [ ] 模型过期检测 → 自动触发重训 pipeline
- [ ] 实盘适配: 真实行情数据源 (非 Qlib 缓存)
- [ ] 滑点模型: 引入基于成交量的滑点估计
