# 量化交易系统 - Claude Code 上下文

> 最近一次大幅修订: 2026-08-30。当日消除了多项回测偏差，历史绩效数字全部
> 下修。若看到与本文件不符的旧数字(如 Sharpe 2.055)，以本文件为准。

## 运行环境 (重要)

- **解释器: 因机器而异，用前先实测**。直接用 `python` 走的是系统 3.13，
  装不了 pyqlib，两台机器都一样。
  - `DESKTOP-5F7SQIQ` (用户 gaoyu): `C:\Users\gaoyu\.conda\envs\qlib\python.exe`
    (3.12.13, qlib 0.9.7 / lightgbm / lark_oapi 齐全)。定时任务用的是这个。
  - 另一台 (用户 `1`): `C:\Program Files\Python312\python.exe` (3.12.7)
  - 2026-09-05 核实: 两条路径互斥 —— 各自在对方机器上都不存在。改任务或
    写脚本前先 `ls` 确认，不要照抄。
- 平台已从 macOS 迁移到 **Windows**，注意三条差异:
  1. `multiprocessing` 只支持 `spawn`，无 `fork`。任何调用 `D.features()` 的
     脚本**必须**有 `if __name__ == '__main__':` 保护，否则子进程重新导入模块
     会无限派生。且**必须写成真实文件执行**，管道传 stdin 同样会失败。
  2. CLI 路径不要硬编码，用 `cli_paths.py` 的 `CLAUDE_BIN` 等。
  3. 定时任务用 Windows 任务计划(schtasks)，不是 launchd。

## 策略说明
- **当前模型**: `M01-LGB-D3v3r-v2602` (D_expand_3v_3r = 扩展窗口, 3月验证, 3月重训)
- **因子集**: alpha158_val (**188** 因子)
- **股票池**: CSI 300 时点成分股 (历史并集 549 只)
  - 2026-09-02 曾切到 CSI 800，当日回退: baostock 数据源故障导致
    cn_data_csi800 从未生成，且 ML 路径的 provider_uri 硬编码为 cn_data_bs，
    只改配置不会真正换源，反而会拿到全 NaN 成分股(沉默失败)。
- **持仓**: 见 `config/signal_config.yaml` 的 `topk` / `n_drop` (以配置为准)
- **调仓**: 每 8 个交易日 (2026-09-04 由 5 日改，依据见配置文件注释)
- **成交价**: **收盘价** (次日收盘, T+1)
- **止损**: 8% | **波动率目标**: 8% 年化
- **资金**: 10 万
- **训练**: 3 个随机种子集成 (LightGBM 原先未设种子，同配置两次训练
  主段 Sharpe 0.876 vs 1.449)，含退化回退(验证集早停杀模型时改用无验证集训练)

### 实测绩效 (10万 / 收盘价 / 含整手与全部交易成本 / TopK16 n_drop4 / vol_target 8%)

> 2026-09-05 全部重测。此前的数字 (主段 1.121 / 段1 1.053) 出自 run_vol_target
> 引擎，且当时 paper_trader 的 vol_target 是失效的 —— 两组数字不可直接比较，
> 以下表为准。

用 `run_phase_test.py` 跑，8 个起始相位。**必须报均值与最差相位，不要报单次回测。**
选用 paper_trader 引擎: 它是被 `reconcile.py` 逐日证明与实盘决策一致的那个
(82 个调仓日 0 分叉)。run_vol_target 未进对账，其数字未必描述实盘策略。

| 区间 | Sharpe 均值 | 标准差 | 最差相位 | 总收益 | 最大回撤 | 超额 |
|---|---|---|---|---|---|---|
| 段1 2022-05~2023-12 (样本外) | 0.856 | 0.239 | 0.473 | +13.89% | -8.80% | **+28.33%** |
| 主段 2024-01~2026-09 | 1.161 | 0.209 | 0.977 | +40.77% | -9.35% | +6.46% |

复现: `python run_phase_test.py --phases 8`(主段) /
`python run_phase_test.py --start 2022-05-04 --end 2023-12-29 --tag 段1 --pred-tag pre2024`

**vs 沪深300**: 熊市极强 (段1 超额 +28.33%)，牛市可能跑输 —— 主段超额均值
只有 +6.46%，**最差相位 -1.35%(跑输指数)**。

**vol_target 的代价要看清**: 主段关掉它时超额 +13.77%、回撤 -10.81%；
开启后超额 +6.46%、回撤 -9.35%。**用一半超额换约 1.5 个百分点的回撤**。
这是 2026-09-05 修好 paper_trader 的 vol_target 之后才第一次看到的真实取舍。

⚠ 一个仍未覆盖的缺口: reconcile.py 只比对买卖清单，**仓位大小不在对账
范围内**。"paper_trader 等于实盘"目前只对选股成立，不对仓位大小成立。

## 凭证管理
- 凭证在 `trading_framework/.env`(已 .gitignore，**不入 git**)。
  模板见 `.env.example`。
- 2026-08-30 曾尝试把真实 `.env` 提交，被 GitHub secret scanning 拦截
  (Lark Application Secret)，已从历史中移除。**不要再尝试提交** —— git
  历史永久，密钥进入后即便删除文件仍可还原。
- 环境变量: `FEISHU_APP_ID_1/2`, `FEISHU_APP_SECRET_1/2`,
  `FEISHU_ALLOWED_OPEN_IDS`(白名单，留空则拒绝所有人), `VPN_RESTART_CMD`

## 项目结构
```
trading_framework/
├── smart_bot.py           # 飞书机器人（ML信号/截图解析/重训）
├── daily_runner.py        # 每日信号推送 (任务 DailyRunner, 18:00)
├── retrain_pipeline.py    # 季度重训 pipeline
├── daemon.py              # 守护进程 (任务 FeishuBot, 登录后自启)
├── send_signal.py         # 发送调仓指令到飞书
├── cli_paths.py           # 外部CLI路径跨平台解析 (CLAUDE_BIN 等)
├── qlib_compat.py         # MLflow 3.x 兼容垫片 (导入即生效)
├── .env                   # 凭证（不入 git）
├── .env.example           # 凭证模板（入 git）
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
├── notifications/         # 系统通知文件夹 (各子系统写入, smart_bot读取)
│   ├── __init__.py            # write_notification() / read_recent()
│   ├── mining/                # 因子挖掘通知 (factor_miner)
│   ├── signal/                # 信号/重训/shadow 通知 (daily_runner)
│   ├── reflection/            # 反思通知 (self_reflect)
│   └── monitor/               # 盘中监控告警 (intraday_monitor)
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
├── self_reflect.py        # 每日自我反思 (23:30, Claude驱动)
├── factor_lab/            # 因子实验室
│   ├── factor_miner.py        # 自动因子挖掘 (每周二/五 22:00, --daily模式)
│   ├── utils.py               # 共享工具 (atomic_json_dump, json_default)
│   ├── signal_generator.py    # ML信号生成器
│   ├── paper_trader.py        # 模拟盘引擎
│   ├── mining/                # 因子挖掘模块
│   │   ├── context.py             # Agent 记忆管理
│   │   ├── hypothesis.py          # Claude 因子假说生成
│   │   ├── validator.py           # 表达式验证
│   │   ├── evaluator.py           # IC/ICIR 评估
│   │   ├── backtest_runner.py     # Rolling 回测对比
│   │   ├── direction_registry.py  # 搜索方向注册表 (跨天累积)
│   │   └── planning_agent.py      # Claude 方向规划
│   ├── quanta/                # QuantaAlpha 演化式挖掘
│   │   ├── config.py              # 挖掘超参数
│   │   ├── trajectory.py          # 轨迹数据结构 + Trace
│   │   ├── factor_pool.py         # 因子池准入控制
│   │   ├── evolution.py           # Mutation/Crossover 演化
│   │   ├── eval_agent.py          # 评估 Agent
│   │   └── ast_dedup.py           # AST 去重
│   ├── mining_results/        # 挖掘结果
│   ├── run_rolling_benchmark.py   # rolling训练 (--test-start/--end/--tag/--deal-price)
│   ├── run_vol_target.py          # 波动率目标+换手限制实验 (当日新增)
│   ├── run_capital_impact.py      # 资金规模影响 (--capitals 10w/100w/1y)
│   └── run_signal_decay_benchmark.py # 信号衰减分析
├── qlib_engine/
│   └── data_setup.py     # BaoStock → Qlib bin 数据
├── data/                  # 行情数据
├── memory/                # 用户记忆（按 open_id 隔离）
└── logs/                  # 日志
```

## 日常工作流

### 每日 18:00 自动推送 (Windows 任务计划 TradingSystem-DailyRunner)
```
daily_runner.py → 刷新数据 → 自动重训检查 → 生成ML信号 → 信号健康检查 → 飞书推送
```
- 自动重训: 模型距 pred_end > 60天时自动触发 (提前1个月)
- 预测突变防护: 新旧 TopK 重叠率 < 50% → 回退旧预测 + 飞书告警
- **信号健康检查 (2026-08-30 新增)**: 模型退化时**拦截推送**并告警。
  三项判据: 预测标准差异常收缩 / 分数区分度不足 / `best_iteration` 过低(<20)。
  起因是 Window 11 的 LightGBM 第 3 轮即早停(阈值 80)，因验证期 2026 Q2
  的 IC 为负，模型学不到东西；3 棵树让 300 只股票只剩 123 个不同分数，
  TopK 里 8 只并列同分 —— 系统原本会把这份随机名单当交易信号推出去。
- 每日记录净值到 `nav_history`，波动率目标靠它估计已实现波动率。

### 盘中监控 (任务 TradingSystem-IntradayMonitor, 9:25~15:05)
```
intraday_monitor.py → Sina实时行情 → 止损/异动/待执行订单 → 飞书推送
```
- 每 5 分钟轮询，11:30~13:00 午休跳过
- 止损阈值: -8% (复用 settings.STOP_LOSS)
- 日内急跌: -5%
- 15:00 收盘总结

### 因子挖掘 (任务 TradingSystem-FactorMiner，**每周二/五 22:00**)

**2026-09-05 由工作日每天改为每周 2 次。** 理由: 一次 session 跑 5 小时、
多轮 LLM 调用，每天跑额度消耗大。而市场每天只新增 1 个交易日，对用数年数据
训练、9 窗口评估的因子而言信息增量约等于零 —— **提高频率不增加信息，只增加
搜索次数，而搜索次数正是过拟合的来源**(这也是新增 DSR/PBO 检验要对付的问题)。

**历史: 2026-08-30 曾暂停** —— 挖掘的评估期与验收测试期原本完全相同
(`quanta/config.py` 的 `ROLLING_EVAL_TEST_*` == `run_rolling_benchmark.TEST_*`)，
等于用考题当练习册 —— 被选中的因子必然在样本内占优。实证: 已接受的 22 个
因子样本内 +7.4%、样本外 **-12.9%**(符号反转)，已全部从 `mined.py` 清除
(归档于 `mined.py.rejected_20260830`)。

已修复: 划出 `MINING_HOLDOUT_START = '2026-02-06'` 边界并加导入时自检
(挖掘评估期 2024-01-01~2026-02-05，严格早于边界)。

2026-09-04 恢复定时任务。安全性依据: 实盘 preset 是 `alpha158_val`，**不含**
`mined` —— 挖掘产物写入 `factors/mined.py` 只会进 `alpha158_val_mined`，
必须经影子验证 + 人工 promote 才可能上线。写入前还有 beat_baseline 与 DSR
两道门。仍需人工确认 holdout 区间的增量后才做 promote。


```
factor_miner.py --daily
  → Phase A: 加载全局 FactorPool + DirectionRegistry
  → Phase B: 广度探索 pending 方向 (Claude假说→构造→验证→IC评估→回测→反馈)
  → Phase C: 深度挖掘 (mutation 已有方向)
  → Phase D: 质量筛选 → 全量 Rolling 回测 → beat_baseline 则写 mined.py + shadow
```
- 三种模式: `--daily` (默认, 每日渐进), `--evolved` (单次全量演化), `--legacy` (旧流程)
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

### 系统通知
- 各子系统在推送飞书的同时写入 `notifications/{category}/{date}.jsonl`
- smart_bot 的 Claude 会话自动注入最近 24 小时的通知摘要
- 通知文件 7 天后自动清理
- 格式: `{"ts": "...", "level": "info|warn|error", "title": "...", "body": "..."}`
- 分类: mining(因子挖掘) / signal(信号/重训/shadow) / reflection(反思) / monitor(盘中监控)

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
**所有命令用**: `"C:\Program Files\Python312\python.exe" -X utf8 ...`

```bash
python daemon.py          # 启动守护进程 (smart_bot)
python daemon.py stop     # 停止
python daemon.py restart  # 重启

# Windows 任务计划 (已创建并启用，"仅在用户登录时运行")
schtasks /Query /TN "TradingSystem-DailyRunner"      # 每日 18:00 信号推送
schtasks /Query /TN "TradingSystem-PaperTrader"      # 工作日 18:30 模拟盘重放
schtasks /Query /TN "TradingSystem-IntradayMonitor"  # 9:25-15:05 每 5 分钟
schtasks /Query /TN "TradingSystem-FactorMiner"      # 每周二/五 22:00 因子挖掘
schtasks /Query /TN "TradingSystem-SelfReflect"      # 每日 23:30 自我反思
schtasks /Query /TN "TradingSystem-Reconcile"        # 每日 19:30 双路径对账
# 五个任务的 WorkingDirectory 必须是 trading_framework —— 2026-09-04 发现
# 原先为空，相对路径会解析到 System32
schtasks /Change /TN "TradingSystem-DailyRunner" /DISABLE   # 临时停用
# 若需未登录也运行，得在任务计划程序 GUI 里改并存储账户密码

python monitor/intraday_monitor.py --dry-run    # 单次检查 (不推送)
python monitor/intraday_monitor.py --once       # 单次检查 + 推送

# 因子挖掘 (工作日 22:00 定时)
python -m factor_lab.factor_miner                       # 每日渐进式 (默认)
python -m factor_lab.factor_miner --daily --smoke-test   # 快速验证 (2方向, 1h)
python -m factor_lab.factor_miner --evolved              # 单次全量演化式
python -m factor_lab.factor_miner --evolved --smoke-test # 演化快速验证
python -m factor_lab.factor_miner --legacy               # 旧流程
python -m factor_lab.factor_miner --dry-run              # 不推送不回测
python -m factor_lab.factor_miner --report               # 查看历史
python -m factor_lab.factor_miner --accept run_001       # 接受因子 + 创建 shadow

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

---

## 研究方法约束 (2026-08-30 血泪总结，务必遵守)

当日有 **4 个"单段最优"的结论在另一段样本上全部翻车**。任何只在一段数据上
验证的结论，默认应假设它是噪声。

| 单段看起来的结论 | 验证后的真相 |
|---|---|
| 22 个挖掘因子 +7.4% | 样本外 **-12.9%**，已清除 |
| vol_target=10% 全期最优 | 样本外跑输基线，改用 8% |
| 每日调仓 | 两段都大幅落后 (0.28 vs 1.41) |
| 3年滑动窗口 +46% | 段1 最差，平均反不如扩展窗口 |

### 单次回测只是一个抽样 (2026-09-04 新增，比上表更根本)

8 日调仓在 2.5 年样本上只有约 **81 次调仓**。起始日错开 1 天(相位 0~7)就是
一组同样合理、但样本不同的结果。实测 8 个相位间的 Sharpe 标准差 **0.11~0.32**，
而候选参数之间的差异只有 **0.1~0.16** —— 差异完全淹没在噪声里。

具体翻车例: 段1 的 TopK=8 曾录得 Sharpe 1.341(相位0)，8 相位均值只有
**0.840**、最差相位 **0.29**。据此认为"8 只在熊市很强"是错的。

已知的三个独立噪声源:
1. **随机种子** — 未设种子时同配置两次训练 0.876 vs 1.449 (已用 3 种子集成压制)
2. **调仓相位** — 见上
3. **集合迭代顺序** — set 差集决定买入顺序，哈希随机化导致 0.318 vs 0.247

**参数选择必须用配对比较**: 同一组相位下算 (候选 − 现行) 的逐样本差值，
报均值、配对 t、胜出样本数。TopK 8→16 就是这样定的(配对 t=2.74, 12/16
相位胜出)；单相位对比给出的是相反结论。

**必须做的三段验证**（预测缓存已备好，见 `results/rolling/predictions/`）:
- 段1 `--pred-tags pre2024` (2022-05~2023-12) —— 参数选择之外的样本
- 主段 `--pred-tags "" seg2pred` (2024-01~2026-08)
- 用 `run_rolling_benchmark --test-start/--test-end --tag` 可指定任意区间；
  **新 tag 必须登记到 `KNOWN_TAGS`**，否则不同测试期结果会混进同一张表。

### 双路径对账 (2026-09-05 新增，先跑这个再谈别的)

    python reconcile.py                 # 最近 120 个交易日
    python reconcile.py --start 2024-01-02 --end 2026-09-04   # 全区间
    退出码 0=一致 / 1=发现分叉 / 2=无法对账

定时任务 `TradingSystem-Reconcile` 每日 19:30 跑，发现分叉推飞书。

**为什么它比再读一遍代码有用**: 本系统反复出现的是同一类缺陷 ——
同一条规则写了两遍，改一处忘另一处。已发生过 5 次:

    n_drop / vol_target   回测有、实盘有、模拟盘没有         2026-09-03
    deal_price            研究用 open、生产用 close          2026-08-30
    _exposure             回测/实盘各一份，回退方向相反       2026-09-05
    调仓判定               回测按交易日、实盘按运行次数        2026-09-05
    pending_orders        推送走 n_drop、存盘走全量换手       2026-09-05

逐条去猜"下一处会在哪分叉"是猜不完的。对账不需要预测:它把回测决策时刻的
输入原样喂给实盘代码路径，比对 sells/buys 是否一致。分叉不管出在代码、
配置还是数据对齐上，都会当场暴露。

**首次运行即报出 3 个真缺陷** (15 个调仓日里 14 个不一致):

1. **实盘 n_drop 退化成"卖掉代码最小的 N 只"** —— get_signal 只返回 TopK
   的 scores，而卖出候选按定义都在 TopK 之外，全部并列 -inf，排序 tiebreak
   落到股票代码上。当初 Sharpe 0.33->1.27 的改动在实盘是随机挑选。
2. **止损在实盘只是一句建议** —— 回测排单次日执行，实盘只在消息里写
   "建议立即卖出"、不进 pending_orders，且原本在"非调仓日"分支里，调仓日
   压根不检查。
3. **回测把止损股卖掉再买回** —— 买入候选排除的是 `持仓 - 待卖` 而非全部
   持仓，占掉一个 free_slot 把该买的挤出去。

修完 82 个调仓日 / 649 交易日零分叉。

**加新引擎、改任何调仓规则之后，必须跑一次对账再提交。**
新的消费方请调用 `live_portfolio.compute_rebalance_orders`，不要自己算。

### 沉默失败是本系统最大的风险来源

当日共发现 5 处"看起来正常工作、实则什么都没做"，比任何策略参数都危险:

1. 对比表用 `glob("*.json")` 无差别加载 —— 表头写新区间、数字是旧的
2. 全部任务失败仍打印漂亮表格且退出码 0
3. 回测器用集合差集决定买入顺序 —— 字符串哈希随机化导致同配置两次结果不同
   (Sharpe 0.318 vs 0.247)。**任何依赖集合迭代顺序的地方都要显式排序**
4. qlib 对缺失字段不报错，返回**全 NaN 列** —— `'$isST' in columns` 判定为
   True，ST 过滤形同虚设。**必须检查 `.notna().any()`**
5. 模型退化(best_iter=3)后照常推送随机名单

6. **飞书主动推送从未生效** (2026-09-04) —— 5 个推送点都读
   `FEISHU_USER_OPEN_ID`，而 `.env` 里这一项是注释掉的(模板里就带 `#`)。
   `push_feishu` 在凭证缺失时只 log 一行然后 return，消息丢弃、不排队、
   退出码仍是 0。`smart_bot` 不受影响(它回复收到的消息，收件人来自消息本身)
   —— "能对话"恰好掩盖了"推不出来"。
   已修: `feishu_target.resolve_open_id()` 回落到 `FEISHU_ALLOWED_OPEN_IDS`
   的第一个，凭证缺失改为落盘补发。**改推送相关代码后必须看返回的
   `resp.success()`，不要拿自己脚本的 print 当投递凭据。**
7. **交易日判断卡死整条链路** (2026-09-04) —— `is_trading_day()` 调
   `baostock.login()`，该接口无超时参数，源挂了就永久阻塞。盘中监控一个
   实例从 09:35 卡到 11:05、烧 1394 秒 CPU、一行日志没写，后续每 5 分钟
   触发全被 0x800710E0 拒绝。任务显示 Running，实际整个上午没监控。
   已抽出 `market_calendar.py` 共用，带硬超时 + fail-open。

8. **第三方库的网络调用没有超时 = 降级分支永远不执行** (2026-09-04)
   一天内在四处踩到同一个坑: `baostock.login()` 和 akshare 内部的
   `requests.get` 都不接受 timeout，源一挂就永久阻塞。**挂起不抛异常**，
   所以调用方写好的 `except` 降级分支根本轮不到执行 —— 代码看起来有兜底，
   实际没有。四处: `intraday_monitor.is_trading_day`(卡 1.5h)、
   `daily_runner.is_trading_day`、`live_portfolio.get_current_prices`(卡 1h+，
   下面就写着"失败时降级到本地缓存")、`data_setup_sina` 逐只下载(卡 8.5h)。
   已修: 统一用 `net_guard.run_with_timeout` / `install_default_request_timeout`。
   **调用任何第三方网络库前，先确认它能不能设超时；不能就用 net_guard 包一层。**
9. **数据重建期间目录是空的** (2026-09-04) —— `data_setup*` 原先直接
   `shutil.rmtree(生产目录)` 再重建，中间数百个 bin 文件的写入窗口里，
   任何读数据的任务(22:00 挖掘、盘中监控、手动回测)拿到的是全 NaN
   而不是报错。已改为建到 `.building` 旁路目录再 `os.replace` 原子替换。
   注: `data_setup.py`(baostock 源) 仍是旧写法，它已不是主源，但若重新启用需一并改。

**写新代码时**: 宁可显式失败，不要静默降级。
**凡是"对外发生效果"的动作(推送/下单/写盘)，必须检查返回值再报成功。**

## 关键设计决策与依据

- **实盘取价用新浪原始价，不用 BaoStock 前复权价** (2026-09-04):
  `cost_price` 来自成交截图解析，是**实际成交价(原始价)**。原先拿
  BaoStock 的 `adjustflag=2` 前复权价去比它，除权日之后会系统性错位 ——
  止损判据 `current/cost-1` 凭空多出或抹掉一截收益。仓位市值、可买手数
  同理都该用原始价。换源同时修的是口径，不只是可用性。
  三级降级: 新浪实时(1.1s/5只) → BaoStock → 当日本地缓存，每级都有硬超时。
- **成交价用 close 不用 open**: one-switch 实验(其余变量固定)显示
  open 超额 27.11% vs close 11.90%，超额腰斩。开盘价是实盘最难兑现的价格。
- **n_drop=2 换手限制**: 原为全量换手，5 日调仓下几乎每次换掉全部 12 只，
  20万 2.5 年约 1734 笔交易，成本吃掉约本金 14%。段1 Sharpe 0.33 → 1.27。
  **这是当日单项影响最大的改动。**
- **保持扩展窗口**: 滑动窗口在一段占优但另一段最差；学术大样本(19 亿观测)
  亦支持扩展窗口。且 `best_iter` 标准差 164，窗口对比本身噪声极大。
- **时点成分股**: 原实现取"今天"的 300 只压平为全历史，丢失期间调出的
  264 只(占历史并集 46.8%)。修复后 Sharpe 2.055 → 1.643。

## 待办 / 进行中

- **~~中证800~~** → 已回退到 CSI 300，理由见上。2026-09-04 用户确认继续用
  CSI 300，相关残留已清理(csi800 成分股缓存、cn_data_bs.broken_249)。
  `data_setup.py --universe csi500/csi800` 的下载能力保留可用。
  **若要重做，先解决 provider_uri 硬编码**，否则换池是沉默失败。
- ST 过滤已实现(日线 `isST`，时点状态无前视)，但**旧的 cn_data_bs 数据集
  没有该字段**，只有新下载的数据集才生效。
- 未实现: "regime-trust gate"(arXiv:2603.13252) —— 训练副模型预测主模型的
  排序误差，用于判断"今天该不该交易"，比当前的硬阈值健康检查更严谨。
