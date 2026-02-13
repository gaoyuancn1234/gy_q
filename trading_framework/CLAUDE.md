# 量化交易系统 - Claude Code 上下文

## 策略说明
- **当前模型**: `M01-LGB-D3v3r-v2602`
  - M01 = 第一版模型
  - LGB = LightGBM
  - D3v3r = D_expand_3v_3r (扩展窗口, 3月验证, 3月重训)
  - v2602 = 2026年02月重训
- **因子集**: alpha158_val (210因子)
- **持仓**: TopK 12-20 等权 (策略A TopK自适应)
- **调仓**: 每5个交易日
- **止损**: 8%
- **资金**: 10万实盘
- **回测**: Sharpe 2.055, MDD -12.45%, Return 111% (2024-2026)

## 凭证管理
- 所有敏感信息存放于 `.env` 文件（已 .gitignore）
- 环境变量: `FEISHU_APP_ID_1/2`, `FEISHU_APP_SECRET_1/2`, `FEISHU_USER_OPEN_ID`

## 项目结构
```
trading_framework/
├── smart_bot.py           # 飞书机器人（ML信号/截图解析/重训）
├── daily_runner.py        # 每日信号推送 (launchd 18:00)
├── retrain_pipeline.py    # 季度重训 pipeline
├── daemon.py              # 守护进程（自动重启）
├── send_signal.py         # 发送调仓指令到飞书
├── .env                   # 敏感凭证（不入 git）
├── config/
│   ├── settings.py        # STRATEGY_TYPE='ml', ML_CONFIG
│   └── signal_config.yaml # ML信号配置 (rolling/preset/topk)
├── portfolio/
│   ├── trade_executor.py  # 调仓指令生成 (rs/qlib/ml 三模式)
│   ├── live_portfolio.py  # 实盘持仓管理 (资金约束+止损)
│   ├── live_holdings.json # 实盘持仓文件
│   └── holdings.json      # RS策略持仓 (旧)
├── monitor/
│   ├── intraday_monitor.py  # 盘中监控 (Sina实时行情, 止损/异动/订单提醒)
│   └── monitor_state.json   # 监控状态 (每日自动重置)
├── shadow_manager.py      # 影子交易验证系统
├── shadow/                # 影子验证数据
│   ├── registry.json          # 候选注册表
│   ├── configs/               # 独立 signal config
│   ├── state/                 # 独立 predictions + rolling_info
│   └── daily_log/             # 每日信号对比 + 汇总
├── news_sentinel.py       # 情绪哨兵 (新闻获取+Claude分析+信号过滤)
├── experiment_manager.py  # 实验盘管理器 (注册/生命周期/绩效)
├── experiment/            # 实验盘数据
│   ├── registry.json          # 实验注册表
│   ├── news_cache/            # 新闻缓存 ({date}_macro.json 等)
│   └── daily_log/             # 每日对比 + 汇总
├── factor_lab/            # 因子实验室
│   ├── factor_miner.py        # 自动因子挖掘 (每周日20:00)
│   ├── signal_generator.py    # ML信号生成器
│   ├── paper_trader.py        # 模拟盘引擎
│   ├── mining/                # 因子挖掘模块
│   │   ├── context.py         # Agent 记忆管理
│   │   ├── hypothesis.py      # Claude 因子假说生成
│   │   ├── validator.py       # 表达式验证
│   │   ├── evaluator.py       # IC/ICIR 评估
│   │   └── backtest_runner.py # Rolling 回测对比
│   ├── mining_results/        # 挖掘结果
│   ├── run_rolling_benchmark.py   # rolling训练
│   └── run_signal_decay_benchmark.py # 信号衰减分析
├── qlib_engine/
│   └── data_setup.py     # BaoStock → Qlib bin 数据
├── data/                  # 行情数据
├── memory/                # 用户记忆（按 open_id 隔离）
└── logs/                  # 日志
```

## 日常工作流

### 每日 18:00 自动推送 (launchd)
```
daily_runner.py → 刷新数据 → 自动重训检查 → 生成ML信号 → 飞书推送
```
- 自动重训: 模型距 pred_end > 60天时自动触发 (提前1个月)
- 预测突变防护: 新旧 TopK 重叠率 < 50% → 回退旧预测 + 飞书告警

### 盘中监控 (9:25~15:05, launchd)
```
intraday_monitor.py → Sina实时行情 → 止损/异动/待执行订单 → 飞书推送
```
- 每 5 分钟轮询，11:30~13:00 午休跳过
- 止损阈值: -8% (复用 settings.STOP_LOSS)
- 日内急跌: -5%
- 15:00 收盘总结

### 每周日 20:00 因子挖掘 (launchd)
```
factor_miner.py → Claude生成假说 → 验证 → IC评估 → 冗余检测 → 回测 → 飞书通知
```
- beat_baseline 时自动写入 mined.py + 创建 shadow 验证
- `--accept run_001` 接受因子并创建 shadow 验证 (不直接上线)

### 影子交易验证 + 反转实验
```
factor_miner 发现更优因子 → 写入 mined.py → 创建 shadow candidate (status=active)
daily_runner 每日: 为每个 active shadow 生成独立信号 → 记录对比 → 飞书附简报
验证期到期 → 飞书推送对比报告 → 人工: promote / reject / extend

promote → 旧基线自动创建反转影子 (status=reverse_shadow, 20天)
         → 自动重训暂停 (有 reverse_shadow 活跃时跳过)
反转到期 → 飞书推送反转实验报告
         → archive: 封存旧模型 (status=archived, 不再重训)
         → rollback: 回退到旧基线 (恢复 config + predictions)
```
- shadow 只记录不执行，不影响实盘
- promote 会备份 live config → 更新 preset → 自动创建反转影子 → 需手动 retrain
- 模型生命周期: active → promoted/rejected | reverse_shadow → archived/rejected

### 实验盘 (情绪哨兵)
```
创建实验 → daily_runner 每日: 获取新闻 → Claude宏观分析 → 个股风险筛查 → 后处理信号 → 记录对比
验证期到期 → 飞书推送对比报告 → 人工: reject / extend
```
- 与 shadow 区别: 实验盘使用**同一基础信号** + 后处理，不需要独立重训
- 数据源: CCTV新闻联播(ak.news_cctv) + 财联社电报(ak.stock_info_global_cls) + 个股新闻(ak.stock_news_em)
- 宏观情绪: bearish=-4 TopK / neutral=0 / bullish=+2 TopK
- 个股过滤: 被调查/ST/诉讼/制裁/退市/违规/业绩暴雷 → 移除并递补
- 生命周期: active → completed/rejected

### 飞书命令
- `信号` - 生成ML调仓信号
- `持仓` - 查看实盘持仓
- `监控` - 盘中监控状态
- `挖掘` - 因子挖掘状态
- `影子` - 影子交易验证状态
- `实验盘` - 实验盘状态
- `重训` - 季度模型重训
- `追加5万` - 追加资金(下次调仓日自动分配)
- `清仓` - 清空所有持仓
- `资金10万` - 设置初始资金(需先清仓)
- 截图 + `已执行` - 解析成交截图更新持仓

### 季度重训
```bash
python retrain_pipeline.py              # 全量 (~5min)
python retrain_pipeline.py --skip-data  # 跳过数据刷新
```

## 启动/管理
```bash
python daemon.py          # 启动守护进程 (smart_bot)
python daemon.py stop     # 停止
python daemon.py restart  # 重启

# launchd (daily_runner)
launchctl load ~/Library/LaunchAgents/com.trading.daily-runner.plist
launchctl unload ~/Library/LaunchAgents/com.trading.daily-runner.plist

# launchd (盘中监控)
launchctl load ~/Library/LaunchAgents/com.trading.intraday-monitor.plist
launchctl unload ~/Library/LaunchAgents/com.trading.intraday-monitor.plist
python monitor/intraday_monitor.py --dry-run    # 单次检查 (不推送)
python monitor/intraday_monitor.py --once       # 单次检查 + 推送

# launchd (因子挖掘)
launchctl load ~/Library/LaunchAgents/com.trading.factor-miner.plist
launchctl unload ~/Library/LaunchAgents/com.trading.factor-miner.plist
python -m factor_lab.factor_miner --dry-run     # 测试运行
python -m factor_lab.factor_miner --report      # 查看历史
python -m factor_lab.factor_miner --accept run_001  # 接受因子 + 创建 shadow

# 影子交易管理
python daily_runner.py --shadow-status              # 查看状态
python daily_runner.py --promote-shadow shadow_001   # 晋升 (自动创建反转影子)
python daily_runner.py --reject-shadow shadow_001    # 拒绝
python daily_runner.py --extend-shadow shadow_001 10 # 延长验证
python daily_runner.py --archive-shadow shadow_002   # 封存 (确认新模型更优)
python daily_runner.py --rollback-shadow shadow_002  # 回退到旧基线

# 实验盘管理
python daily_runner.py --experiment-status               # 查看实验盘状态
python daily_runner.py --create-sentiment-experiment      # 创建情绪哨兵实验
python daily_runner.py --reject-experiment exp_001        # 终止实验
python daily_runner.py --extend-experiment exp_001 15     # 延长实验

# 情绪哨兵测试
python news_sentinel.py --test-macro                     # 测试宏观新闻获取
python news_sentinel.py --test-stocks SH600036 SH601318  # 测试个股新闻
```
