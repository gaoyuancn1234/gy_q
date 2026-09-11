# CSI2000 当前状态

> 写于 2026-09-11。本目录是中证2000 研究树，不是沪深300 实盘。
> 数字、参数、成交时钟以本文和 `config/signal_config.yaml` 为准。
> `CLAUDE.md` 里大段仍是 CSI300 历史（Sharpe 1.205、TopK16、止损 8%、
> vol_target 8%），**那些不是 CSI2000 的约束**。研究方法（8 相位、
> 对账、禁止预测 A 交易 B）仍然适用。

## 2026-09-11 18:00 现场

不是卡死在出分/下单，是**隔夜重训没把 pkl 写出来，后面全在空等**。

当时情况：

- 07:43 起跑 `alpha158_ovn` 11 窗滚动重训。约 10:21 进入 Window 10/11。
- `python -m daily wait_live` 从上午一直等到约 17:53，日志只有
  `ovn pkl 未到`。
- 18:00 复查：本机**没有 Python 进程**；
  `D_expand_3v_3r_alpha158_ovn_LightGBM.pkl` **不存在**；
  yaml 仍是 `preset: alpha158` / `exec_lag: 1`；
  掘金仿真 `21761e6d-ad82-11f1-8afe-00163e022aa6` 仍是 10 万空仓
  （10:02 `--check`）。今日 14:45 / 14:50 **没有出分、没有下单**。
- 重训终端和 wait_live 终端已被清掉，最后一次看到的进度停在第 10 窗开头，
  没有「预测已保存」，也没有 traceback。进程是后来自己没了，不是我们主动杀的。

代码侧已经接上、但今天没用上的：

- 唯一入口 `python -m daily`（status/refresh/extend/signal/run/auction/
  prefetch/place/sync/activate_ovn/wait_live）
- `exec_lag` 贯穿 paper_trader / live_portfolio / daily_runner /
  gm_bridge / reconcile
- `persist_boosters` + `auction_infer` 给 14:45 截断出分
- `setup_tasks.ps1` 在 `csi-2000` 目录会 exit 2，防撞 CSI300 任务
- `signal_today.py` / `send_signal --signal` 已拒绝

## 目标（未完成，不要缩小）

把 csi-2000 建成可每日例行、标签=成交=时钟一致的系统：

1. 隔夜模型落地（`alpha158_ovn` pkl + 最后一窗 booster）
2. 一条日更入口（`python -m daily`）
3. `exec_lag` 贯穿回测 / 实盘 / 对账
4. 14:45 出分能跑通，14:50 收盘竞价成交
5. 删掉会误发 CSI300 信号或写错数据源的冗余入口
6. 不碰 CSI300 活系统
7. 用掘金仿真账户 `21761e6d-ad82-11f1-8afe-00163e022aa6`（10 万）
   真正下单建仓

下一交易日：重跑隔夜滚动 → persist → `activate_ovn` → 14:31 prefetch →
14:45 auction → 14:50 place → 15:05 sync。
pkl 落地前不要切 yaml。

## 不要动

- `gy_q/trading_framework`、持仓、定时任务、`cn_data_bs`
- 不要往 CSI2000 中台写新浪/BaoStock
- 不要 `--commit` 覆盖 `cn_data_csi2000`，除非明确要换供给层
- 不要杀原树里可能还在写的分钟缓存

## 这棵树是什么

机构框架（时点成分、截面排序、宽仓、换手上限、涨跌停/整手、成本）
+ 散户容量（10 万、中证2000 小票可成交）。

不做市值中性。成交目标是收盘竞价，不是开盘抢单。

Python（这台机器）: `C:\Program Files\Python312\python.exe`，命令加 `-X utf8`。
Windows 下凡调用 `D.features()` 的脚本必须有 `if __name__ == '__main__'`。

## 数据

| 层 | 位置 |
|---|---|
| 日线原始 | 原树 `tushare_csi2000` 缓存，中台 junction 到 `~/.qlib/data_hub/raw/tushare/daily` |
| Qlib 供给 | `~/.qlib/qlib_data/cn_data_csi2000`（日历约 2019-10-08 ~ 2026-09-09） |
| 15 分钟 | 原树 `minute_csi2000`（2921/2955；缺的多为北交所，不再死循环下） |
| 成分股 | `data/csi2000_membership.json`（2024-02 前用最早快照回填，训练期有幸存者偏差，回测主段不受影响） |

中台只走 Tushare。15000 积分**不含盘中实时价**。日线要收盘后才齐。

```
python -m data_hub status
python -m data_hub validate
```

## 现行配置（以 yaml 为准）

`instruments: csi2000`，`preset: alpha158`，资金 10 万。

- `topk: 100` / `n_drop: 20`（名义 100，10 万能建约 55~62 仓）
- `adaptive_strategy: none`，`vol_target: null`，`stop_loss: null`
- `min_amount: 2000000`，`rebalance_every: 8`，`deal_price: close`
- `exec_lag: 1`（仍是 T 日信号 → T+1 收盘成交）
- `n_seeds: 3`，`qlib_kernels: 2`

旧预测: `factor_lab/results/rolling/predictions/D_expand_3v_3r_alpha158_LightGBM.pkl`

## 已经成立的诊断

1. 模型 IC 强，但 alpha 堆在 T+1 涨停买不进的票上。T 收盘信号、T+1 收盘成交会错过隔夜。
2. A 股隔夜均值偏负。拼的是截面右尾和摩擦，不是「预测更准」。
3. 宽仓单相位（旧标签、T+1 收盘）大约 +50.6% / Sharpe 0.70。**这是一个抽样，不是结论。**
4. 14:45 vs 收盘价格秩相关 1.000；Alpha158 截面秩相关 0.989；ROC5 Top100 重叠 96%。涨停识别不在最后 15 分钟。
5. 路线 B = **14:45 出信号 + 14:50 收盘竞价成交**。不能在 14:45 成交。标签必须和成交同一天。

## 路线 B（进行中）

目标标签: `Ref($open,-1)/$close-1`（T 收盘 → T+1 开盘）。当日已涨停样本置空。
预设名: `alpha158_ovn`。回测成交: `exec_lag: 0`。

2026-09-11 正在跑:

```
python -m factor_lab.run_rolling_benchmark --configs D_expand_3v_3r --presets alpha158_ovn --models LightGBM --test-start 2024-01-01 --test-end 2026-09-09 --force
```

产物（尚未落地）: `D_expand_3v_3r_alpha158_ovn_LightGBM.pkl`

pkl 落地前**不要**改 `preset` / `exec_lag`。落地后只准用 `paper_trader` 回放，不要看 rolling JSON 里的 Qlib 自带回测。

## 还没闭环

- 隔夜重训未完成；配置未切 (`preset`/`exec_lag` 仍是旧口径)
- pkl 落地后需要: 切 yaml → 临时目录 `paper_trader` 回放 → `python -m factor_lab.persist_boosters` 落盘最后一窗
- CSI2000 宽仓未做 8 相位；段1（2022–2023）成分是回填的，不能当参数依据
- 训练用的是完整日线，不是 14:45 截断日线（特征很近，但对实盘仍有一点偷看）
- 定时任务脚本默认只打印。确认后再 `setup_csi2000_tasks.ps1 -Apply`，不要跑 `setup_tasks.ps1`

## 数字怎么报

- 必须报 8 相位均值和最差相位，不要报单次回测
- 引擎只用 `paper_trader`（对账过的那条）。`run_vol_target` / Qlib `backtest_daily` 不是实盘数字
- 改调仓规则之后先跑 `reconcile.py`
- 标签、成交价、下单时点必须同一套。只改下单时间等于预测 A 交易 B

## 每日例行

唯一入口 (在 `csi-2000` 目录):

```
python -X utf8 -m daily status
python -X utf8 -m daily prefetch     # 14:00 预取当日分钟
python -X utf8 -m daily auction      # 14:45 截断出分写 pending (exec_lag=0)
python -X utf8 -m daily run          # 15:05 后: 刷新日线 + 增量预测
                                     # exec_lag=0 时不再改写当天 pending
python -X utf8 -m factor_lab.persist_boosters   # 训完后落盘最后一窗 booster
```

`exec_lag` 已贯穿 `paper_trader` / `live_portfolio` / `daily_runner` /
`gm_bridge` / `reconcile`。下单日看 `holdings['execute_on']`，时间窗 14:50~15:05。

CSI2000 掘金仿真账户 `21761e6d-ad82-11f1-8afe-00163e022aa6`（10 万、空仓），
不要用 CSI1000/CSI300 那个账户。预测完成后:

```
python -X utf8 gm_bridge.py --check
python -X utf8 gm_bridge.py --place
python -X utf8 gm_bridge.py --sync
```

定时任务脚本 `setup_csi2000_tasks.ps1` 默认只打印、不注册。
**不要**在本目录跑 `setup_tasks.ps1` (那是 CSI300 的 TradingSystem-*)。

不要用 `signal_today.py` / `send_signal --signal` (RS 沪深300 蓝筹)。

## 常用命令

```
python -X utf8 run_core.py
python -X utf8 -m factor_lab.paper_trader replay
python -X utf8 run_phase_test.py --phases 8
python -X utf8 reconcile.py
```

解释器必须是上面那条 Python 3.12，不要用系统 `python`。
