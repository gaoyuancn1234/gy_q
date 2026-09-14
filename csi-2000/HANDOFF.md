# 交接说明（2026-09-13 写，换机器时先读这份）

## 新机器上先做三件事

1. **解释器路径因机器而异，先 `ls` 确认**，不要照抄
   - `DESKTOP-5F7SQIQ`(用户 gaoyu)：`C:\Users\gaoyu\.conda\envs\qlib\python.exe`
   - 另一台(用户 `1`)：`C:\Program Files\Python312\python.exe`
   - 两条路径互斥，各自在对方机器上都不存在

2. **拷 `.env`** —— 不入 git（GitHub secret scanning 会拦，2026-08-30 试过）。
   从本机 `csi-2000/.env` 直接复制。需要的键见 `.env.example`：
   `FEISHU_APP_ID_1` / `FEISHU_APP_SECRET_1` / `FEISHU_ALLOWED_OPEN_IDS` /
   `GM_TOKEN` / `GM_ACCOUNT_ID` / `GM_STRATEGY_ID` / `TUSHARE_TOKEN`

3. **数据不在 git 里**，需要重新拉：
   ```
   ~/.qlib/qlib_data/cn_data_csi2000/          Qlib bin（日历 2019-10-08~2026-09-11）
   ~/.qlib/data_hub/raw/tushare/daily/         日线 parquet 1685 个
   csi-2000/factor_lab/results/.cache/minute_csi2000/   分钟 304 只（随机样本）
   ```
   重建：`python -m data_hub refresh`（全量 Tushare，小时级）

   **落盘模型也不在 git**（`factor_lab/results/rolling/models/`，19M，
   且没有 qlib bin 数据时用不上）。数据到位后：
   `python retrain_pipeline.py` 然后 `python -m factor_lab.persist_boosters`

## 落地前先跑这两个

```
python smoke.py                                        # 11 项，约 5 分钟
python reconcile.py --start 2024-01-02 --end 2026-09-11  # 全区间，约 10 分钟
```

2026-09-14 两项均已通过（对账 654 交易日 / 82 调仓日 / 分叉 0）。
两个退出码都必须 0。`smoke` 验"路径能不能跑"，`reconcile` 验"两条路径结果是否一致"。
**短窗对账会漏掉挂单积压类问题**（60 天全过、全区间报 54 个），所以用全区间。

## 当前状态

### 配置（五项都过了 8 相位配对检验，不要随便改）

```
preset=alpha158_ovn   exec_lag=0   instruments=csi2000
topk=100  n_drop=20  rebalance_every=8
vol_target=0.25  vol_unknown_exposure=0.40  stop_loss=null
```

### 业绩（主段 2024-01-02~2026-09-11，8 相位，基准中证2000 +27.96%）

| 口径 | Sharpe | 最差相位 | 回撤 | 超额 |
|---|---|---|---|---|
| exec_lag=1（干净，无前视） | 0.977 | 0.789 | -19.68% | +40.81% |
| exec_lag=0（生产） | 1.240 | 1.011 | -18.42% | +70.35% |

生产口径含约 7% 前视虚高（实测，非估计），折回约 +65%。年化约 +21.5%/年。

### 定时任务（已在本机注册，新机器要重新装）

```
powershell -ExecutionPolicy Bypass -File .\setup_csi2000_tasks.ps1 -Apply
```

| 时间 | 任务 | 动作 |
|---|---|---|
| 14:00 | CSI2000-Prefetch | 取 2000 只当日分钟（实测约 20 分钟） |
| 14:46 | CSI2000-Auction | 14:45 截断特征出分，写 pending |
| **14:50** | **CSI2000-Place** | **向掘金仿真账户真实下单** |
| 15:05 | CSI2000-Sync | 回读成交，写持仓与净值 |
| 19:30 | CSI2000-Reconcile | 全区间对账 |

**两台机器不要同时装**，否则会重复下单。

## 未完成的事（按优先级）

1. **10 万本金买不出 100 只票** —— 当前最大的回测/实盘口径分叉。
   A股一手 100 股，中证2000 中位股价 13.48 元 → 一手 1,348 元。
   首日敞口 40% → 预算 4 万 → 等分到 100 只每只 400 元，一手都不够。
   实测(09-11 真实价，每档抽 300 次取中位)：

   | 配置 topk | 实建仓 | 每仓均值 |
   |---|---|---|
   | 100(当前) | **17** | 2,142 |
   | 40 | 23 | 1,364 |
   | 20 | 16 | 2,085 |
   | 10 | 10 | 3,464 |

   要在 topk=100 下建满：敞口 100% 需约 50 万、敞口 40% 需约 120 万。
   **而回测不受此约束** —— qlib 缺 `$factor`，`trade_unit=100` 没生效
   (`run_rolling_benchmark.py` 里有原注释)。所以 Sharpe 1.240 对应真·100 只
   等权，实盘 10 万只拿得到"top100 里恰好便宜的 17 只"= 叠了个低价因子。
   `max_topk_by_capital` 就是为此准备的杠杆，现设 100 不起作用。
   **选定方案(加资金 / 降 topk)后必须重跑 8 相位配对再上。**

2. **三项参数待重测** —— 都是 `exec_lag` 坏掉时定的：
   `vol_target` / `vol_unknown_exposure` / `adaptive_strategy`
3. **因子挖掘没正式跑过一轮**（提示词刚修好）
4. **标签路线(b)：8 日前向标签重训** —— "标签与持仓期错配"里唯一没试过的
5. **`$isST` 全 NaN** —— 该从 Tushare 补（`data_hub/assemble.py:78` 写死 NaN）
6. **卖出候选无分数占比随时间从 17 涨到 42**（分母约 95）。
   是买入后离开中证2000 的票不再有预测，被"视为最差"优先卖出。
   两引擎一致(分叉 0)，不是缺陷，但意味着 n_drop=20 的名额相当一部分被
   "已离开指数"消耗，模型卖出信号起作用的余地比看上去小。
   **往哪边偏没测过** —— 离开指数既可能是跌没了，也可能是涨大了被调走。

## 运维约束（踩出来的）

- **两个 qlib 重任务不能并发** —— joblib memmap 临时目录互删，
  后启动的死于退出码 127，表现是"莫名其妙失败"没有明确报错。
  `run_phase_test` / `run_param_sweep` / `run_rolling_benchmark` / `reconcile` 必须串行。
- 调 `D.features()` 的脚本**必须**有 `if __name__ == '__main__':`（Windows 只有 spawn）
- 不要在 CSI300 树（`qlib/trading_framework`）里启用 Tushare ——
  `raw/tushare/daily` 不按股票池分目录，会互相覆盖。已加 `.owner` 标记会报错。

## 研究纪律（CLAUDE.md 有完整血泪记录）

- 单段结论默认是噪声，参数选择必须 8 相位配对，报 t 值和胜出数
- **耦合参数必须一起扫** —— `--param topk+n_drop`。只扫 topk 而 n_drop 固定，
  测到的是换手成本不是 topk 效应（我踩过）
- 宁可显式失败，不要静默降级
- 凡是对外发生效果的动作（推送/下单/写盘），必须检查返回值再报成功

## 三条自我约束（2026-09-13 定，因为我违反过每一条）

1. 关于数据的结论**必须看分布不能看单点** —— "这一天是这样"推不出"一直是这样"
   （我看了 1 天的量纲就改了全历史 ×100，实际只有 2/1675 天是那样）
2. **参数建议必须先过配对检验再说出口**（我按排名分桶就建议 topk 降到 30，
   配对检验是 0/8 胜出、超额 -38.36%）
3. **不在两个测量之间插值**，说清楚哪个测了哪个没测
   （我说过"真值在两者之间"，没有任何依据）

## 详细记录

`doc/progress_20260912.md` —— 含全部实测数字、每处更正的原因、被推翻的判断。
