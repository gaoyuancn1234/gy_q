# 量化交易系统 - Claude Code 上下文

> 最近一次大幅修订: 2026-08-30。当日消除了多项回测偏差，历史绩效数字全部
> 下修。若看到与本文件不符的旧数字(如 Sharpe 2.055)，以本文件为准。

## 运行环境 (重要)

- **必须使用 conda 环境**: `C:\Users\gaoyu\.conda\envs\qlib\python.exe` (Python 3.12)
  系统 Python 3.13 装不了 pyqlib，直接用 `python` 会失败。
- 平台已从 macOS 迁移到 **Windows**，注意三条差异:
  1. `multiprocessing` 只支持 `spawn`，无 `fork`。任何调用 `D.features()` 的
     脚本**必须**有 `if __name__ == '__main__':` 保护，否则子进程重新导入模块
     会无限派生。且**必须写成真实文件执行**，管道传 stdin 同样会失败。
  2. CLI 路径不要硬编码，用 `cli_paths.py` 的 `CLAUDE_BIN` 等。
  3. 定时任务用 Windows 任务计划(schtasks)，不是 launchd。

## 策略说明
- **当前模型**: `M01-LGB-D3v3r-v2602` (D_expand_3v_3r = 扩展窗口, 3月验证, 3月重训)
- **因子集**: alpha158_val (**188** 因子)
- **股票池**: 沪深300 (时点成分股，历史并集 564 只)
- **持仓**: TopK 12 等权，**n_drop=2** (每次调仓最多换 2 只)
- **调仓**: 每 5 个交易日
- **成交价**: **收盘价** (次日收盘, T+1)
- **止损**: 8% | **波动率目标**: 8% 年化 (敞口 0.2~1.0，不加杠杆)
- **资金**: 20 万

### 实测绩效 (20万 / 收盘价 / 含整手与全部交易成本)
| 区间 | Sharpe | 收益 | 最大回撤 |
|---|---|---|---|
| 段1 2022-05~2023-12 (样本外) | 1.272 | 20.47% | -5.93% |
| 2024-01~2026-08 | 1.162 | 36.92% | -11.48% |
| 其中 2026-02~08 (最差) | 0.674 | 3.75% | -6.82% |

**vs 沪深300**: 五段里赢四段。熊市极强(段1 超额 +38.68%)，牛市可能跑输
(2025 年 -7.42%)。全期超额仅 +2.41% —— 价值主要在**更小回撤**，不在更高收益。

## 凭证管理
- 凭证在 `trading_framework/.env`。
- **注意: 2026-08-30 起 `.env` 已按用户要求提交进 git**(私有仓库
  github.com/gaoyuancn1234/gy_q)。若仓库转为 Public / 加协作者 / 被 fork，
  需去 open.feishu.cn 重置应用凭证。
- 环境变量: `FEISHU_APP_ID_1/2`, `FEISHU_APP_SECRET_1/2`,
  `FEISHU_ALLOWED_OPEN_IDS`(白名单，留空则拒绝所有人), `VPN_RESTART_CMD`

## 项目结构
```
trading_framework/
├── smart_bot.py           # 飞书机器人（ML信号/截图解析/重训）
├── daily_runner.py        # 每日信号推送 (launchd 18:00)
├── retrain_pipeline.py    # 季度重训 pipeline
├── daemon.py              # 守护进程（自动重启）
├── send_signal.py         # 发送调仓指令到飞书
├── cli_paths.py           # 外部CLI路径跨平台解析 (CLAUDE_BIN 等)
├── qlib_compat.py         # MLflow 3.x 兼容垫片 (导入即生效)
├── .env                   # 凭证（2026-08-30 起已入 git，见"凭证管理"）
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
│   ├── factor_miner.py        # 自动因子挖掘 (每日22:00, --daily模式)
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

### 盘中监控 (9:25~15:05, launchd)
```
intraday_monitor.py → Sina实时行情 → 止损/异动/待执行订单 → 飞书推送
```
- 每 5 分钟轮询，11:30~13:00 午休跳过
- 止损阈值: -8% (复用 settings.STOP_LOSS)
- 日内急跌: -5%
- 15:00 收盘总结

### 每日 22:00 因子挖掘 (⚠ 当前已暂停，未配置定时任务)

**暂停原因 (2026-08-30)**: 挖掘的评估期与验收测试期原本完全相同
(`quanta/config.py` 的 `ROLLING_EVAL_TEST_*` == `run_rolling_benchmark.TEST_*`)，
等于用考题当练习册 —— 被选中的因子必然在样本内占优。实证: 已接受的 22 个
因子样本内 +7.4%、样本外 **-12.9%**(符号反转)，已全部从 `mined.py` 清除
(归档于 `mined.py.rejected_20260830`)。

已修复: 划出 `MINING_HOLDOUT_START = '2026-02-06'` 边界并加导入时自检。
**重新启用前必须**: 在 holdout 区间验证挖掘产出确实有增量。


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
**所有命令用 conda 环境的 python**: `C:\Users\gaoyu\.conda\envs\qlib\python.exe -X utf8 ...`

```bash
python daemon.py          # 启动守护进程 (smart_bot)
python daemon.py stop     # 停止
python daemon.py restart  # 重启

# Windows 任务计划 (已创建并启用，"仅在用户登录时运行")
schtasks /Query /TN "TradingSystem-DailyRunner"      # 每日 18:00
schtasks /Query /TN "TradingSystem-IntradayMonitor"  # 9:25-15:05 每 5 分钟
schtasks /Query /TN "TradingSystem-SelfReflect"      # 每日 23:30
schtasks /Change /TN "TradingSystem-DailyRunner" /DISABLE   # 临时停用
# 若需未登录也运行，得在任务计划程序 GUI 里改并存储账户密码

python monitor/intraday_monitor.py --dry-run    # 单次检查 (不推送)
python monitor/intraday_monitor.py --once       # 单次检查 + 推送

# 因子挖掘 (当前暂停，无定时任务)
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

**必须做的三段验证**（预测缓存已备好，见 `results/rolling/predictions/`）:
- 段1 `--pred-tags pre2024` (2022-05~2023-12) —— 参数选择之外的样本
- 主段 `--pred-tags "" seg2pred` (2024-01~2026-08)
- 用 `run_rolling_benchmark --test-start/--test-end --tag` 可指定任意区间；
  **新 tag 必须登记到 `KNOWN_TAGS`**，否则不同测试期结果会混进同一张表。

### 沉默失败是本系统最大的风险来源

当日共发现 5 处"看起来正常工作、实则什么都没做"，比任何策略参数都危险:

1. 对比表用 `glob("*.json")` 无差别加载 —— 表头写新区间、数字是旧的
2. 全部任务失败仍打印漂亮表格且退出码 0
3. 回测器用集合差集决定买入顺序 —— 字符串哈希随机化导致同配置两次结果不同
   (Sharpe 0.318 vs 0.247)。**任何依赖集合迭代顺序的地方都要显式排序**
4. qlib 对缺失字段不报错，返回**全 NaN 列** —— `'$isST' in columns` 判定为
   True，ST 过滤形同虚设。**必须检查 `.notna().any()`**
5. 模型退化(best_iter=3)后照常推送随机名单

**写新代码时**: 宁可显式失败，不要静默降级。

## 关键设计决策与依据

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

- **中证800 数据下载未完成** (`--universe csi800`，历史并集 1403 只，
  目标目录 `~/.qlib/qlib_data/cn_data_csi800`)。用户方向是中小盘以求更高收益。
  数据齐后需: 三段验证 + **重新校准 vol_target**(8% 是按沪深300 波动定的，
  中小盘波动 15~25%，继续用 8% 会把敞口长期压到 40% 以下)。
- ST 过滤已实现(日线 `isST`，时点状态无前视)，但**旧的 cn_data_bs 数据集
  没有该字段**，只有新下载的数据集才生效。
- 未实现: "regime-trust gate"(arXiv:2603.13252) —— 训练副模型预测主模型的
  排序误差，用于判断"今天该不该交易"，比当前的硬阈值健康检查更严谨。
