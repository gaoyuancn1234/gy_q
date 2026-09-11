# 新机器部署 (Windows)

按顺序做完这七步。每步都有验证命令 —— **不要跳过验证**，本项目反复出现
"看起来跑成功、实则什么都没做"的情况，验证是唯一可靠的判据。

---

## 0. 前置

- Windows 10/11
- Git
- **Python 3.10~3.12**。系统自带的 3.13 装不了 pyqlib(pip 报 no matching
  distribution)。推荐 conda 建独立环境。

```powershell
conda create -n qlib python=3.12 -y
conda activate qlib
```

---

## 1. 拉代码

```powershell
git clone https://github.com/gaoyuancn1234/gy_q.git
cd gy_q
git checkout feature/qlib-integration
```

---

## 2. 装依赖

```powershell
cd trading_framework
python -m pip install -r requirements.txt
```

**验证**(必须全部有版本号，不能只看 pip 没报错):

```powershell
python -c "import qlib, lightgbm, pandas, akshare, baostock, lark_oapi; print(qlib.__version__, lightgbm.__version__, pandas.__version__)"
```

> ⚠ 包名是 `pyqlib` 不是 `qlib`。仓库根目录恰好也叫 qlib，`import qlib`
> 会被目录当成命名空间包冒充成功 —— 装没装上很难看出来，所以必须打印版本号。

---

## 3. 填凭证

**复制 `.env.example` 为 `.env`，填入真实值。**

```powershell
copy .env.example .env
notepad .env
```

`.env` 已在 `.gitignore` 中，**永远不要提交**。2026-08-30 曾尝试提交真实
`.env`，被 GitHub secret scanning 拦截；git 历史是永久的，密钥一旦进入
即便删除文件也能还原。

必填项:

| 键 | 从哪来 | 不填的后果 |
|---|---|---|
| `FEISHU_APP_ID_1` / `FEISHU_APP_SECRET_1` | open.feishu.cn 建企业自建应用 | 机器人无法启动 |
| `FEISHU_ALLOWED_OPEN_IDS` | 先启动机器人 → 发条消息 → 控制台打印你的 open_id | **留空则拒绝所有人**(这是故意的安全默认) |

选填项:

| 键 | 用途 |
|---|---|
| `GM_TOKEN` / `GM_ACCOUNT_ID` / `GM_STRATEGY_ID` | 掘金仿真下单 (myquant.cn，免费不用开户)。不填则 `gm_bridge.py` 不可用，其余功能不受影响 |
| `REDIS_URL` | 临时文件存储。不填会优雅降级到磁盘 |
| `VPN_RESTART_CMD` | 网络中断时尝试恢复 |
| `CLAUDE_BIN` | Claude CLI 路径。默认自动探测，探测失败才需手填 |

> `FEISHU_ALLOWED_OPEN_IDS` 为什么必须配: 机器人收到消息后会以
> `--dangerously-skip-permissions` 调用 Claude CLI，等于把本机的任意命令
> 执行权交给发消息的人。

---

## 4. 下载行情数据 (约 10 分钟)

**数据不在 git 里**(几百 MB 的二进制)，必须在新机器上重新下载。

```powershell
python -m qlib_engine.data_setup_sina
```

**验证**:

```powershell
python -c "import qlib; from qlib.constant import REG_CN; from pathlib import Path; qlib.init(provider_uri=str(Path.home()/'.qlib/qlib_data/cn_data_bs'), region=REG_CN); from qlib.data import D; d=D.features(['SH600036'],['$close'],start_time='2026-08-25',end_time='2026-09-04'); print(f'{len(d)} 行, 末值 {d[\"$close\"].iloc[-1]}')"
```

应输出约 9 行且末值是个正常股价。**行数为 0 或全 NaN 都说明没下成功** ——
qlib 对缺失字段不报错、返回全 NaN 列，所以必须看具体数值。

---

## 5. 注入估值数据 (可选，基本面因子需要)

不做这步，`alpha158_val` 是 188 因子(纯量价)；做完是 210 因子(+22 个估值)。

```powershell
python -m factor_lab.data.akshare_valuation      # 下载，约 40 分钟
python -m factor_lab.data.inject_valuation       # 注入 + 验证
```

末尾必须打印 `✓ 22 个基本面因子可用`。

> 注入的字段会被后续的数据刷新保留(`data_setup_sina._carry_over_injected`)。
> 2026-09-06 之前不会 —— 每次刷新都静默抹掉，而可用性检查又永远报"缺失"，
> 两个 bug 互相掩盖。

---

## 6. 建定时任务

```powershell
powershell -ExecutionPolicy Bypass -File setup_tasks.ps1
```

会创建 7 个任务并**逐个回查确认**。参数:

```powershell
.\setup_tasks.ps1 -List                          # 查看状态与上次结果
.\setup_tasks.ps1 -Remove                        # 全部删除
.\setup_tasks.ps1 -PythonExe C:\path\python.exe  # 指定解释器
```

| 任务 | 时间 | 做什么 |
|---|---|---|
| DailyRunner | 每日 18:00 | 刷新数据 → 生成信号 → 健康/时效检查 → 推飞书 |
| PaperTrader | 工作日 18:30 | 模拟盘重放 |
| Reconcile | 每日 19:30 | 双路径对账，发现分叉推飞书 |
| FactorMiner | 周二/五 22:00 | 因子挖掘 |
| SelfReflect | 每日 23:30 | 自我反思 |
| IntradayMonitor | 9:25~15:05 每 5 分钟 | 盘中监控 |
| FeishuBot | 登录后自启 | 飞书机器人守护进程 |

任务默认"仅在用户登录时运行"。若要未登录也跑，需在任务计划程序 GUI 里改为
"不管用户是否登录都运行"并存储账户密码 —— 脚本无法代劳(要密码)。

---

## 7. 端到端验证

**按顺序跑，每步都要看输出，不要只看退出码。**

```powershell
# 数据源健康 (四条降级路径)
python check_datasources.py

# 双路径对账 —— 必须 0 分叉
python reconcile.py --days 120

# 完整链路演练 (不推送、不下单)
python daily_runner.py --dry-run --force
```

三条都过才算装好。`reconcile.py` 是最重要的一条:它比对实盘代码路径与回测
引擎在同一份输入下的买卖清单与股数，**分叉不管出在代码、配置还是数据对齐
上都会当场暴露**。

---

## 常见问题

**`import qlib` 成功但功能不对** —— 多半是装成了 PyPI 上另一个叫 `qlib`
的包，或者根本没装、被同名目录冒充。用第 2 步的验证命令打印版本号。

**任务显示 Ready 但从未运行(上次运行 1999/11/30)** —— 正常，表示还没到
第一次触发时间。用 `-List` 看 NextRunTime 确认。

**任务跑了但什么也没发生** —— 先查 `WorkingDirectory`。2026-09-04 发现
该项为空时，所有相对路径都解析到 `System32`。`setup_tasks.ps1` 已显式写死。

**飞书收不到推送** —— 检查 `FEISHU_ALLOWED_OPEN_IDS` 是否为空。留空会
拒绝所有人，且推送失败时只 log 一行、退出码仍是 0。

**`.ps1` 报语法错误** —— Windows PowerShell 5.1 读 `.ps1` 默认按 ANSI，
含中文的脚本必须存成 **UTF-8 with BOM**，否则中文注释会被拆坏。

---

## 两台机器之间共享什么

| | 共享 | 说明 |
|---|---|---|
| 代码 | ✓ git | |
| 凭证 | ✗ | 各自填 `.env`。飞书应用可共用，白名单要各自加 |
| 行情数据 | ✗ | 各自下载(第 4 步)。也可直接拷贝 `~/.qlib/qlib_data/cn_data_bs` |
| 预测缓存 | ✗ | `factor_lab/results/rolling/predictions/*.pkl` 不在 git(每个几 MB)。要么拷贝，要么重训 |
| 实盘持仓 | ✗ | `portfolio/live_holdings.json` 不在 git。**两台机器不要同时跑 DailyRunner**，否则持仓状态会打架 |

> ⚠ 最后一条要当心: `live_holdings.json` 是单机状态文件，没有任何并发保护。
> 两台机器同时跑定时任务会各写各的，调仓相位和持仓记录都会分叉。
> 新机器如果只是用来做研究/回测，把 DailyRunner / PaperTrader /
> IntradayMonitor 三个任务停用即可。
