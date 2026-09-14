# CSI2000 专用定时任务。名称带 CSI2000- 前缀, 不会覆盖 CSI300 的 TradingSystem-*。
#
# 默认只列出计划, 不注册。确认机器上没有 CSI300 同名冲突后再:
#     powershell -ExecutionPolicy Bypass -File setup_csi2000_tasks.ps1 -Apply
#
# 时钟:
#   14:00  prefetch  当日分钟打底 (给 14:45 出分)
#   14:25  prefetch2 二次全量, 把截断时点推近 14:45
#   14:46  auction   路线 B 截断出分 (模型未落盘会失败退出, 不误下单)
#
# 为什么要跑两遍 (2026-09-14):
#   stk_mins 每次只返回"到此刻为止"的K, 且必须逐只查(实测 0.47s/只,
#   2000 只约 16 分钟)。所以 14:00 那遍拿到的是 14:00~14:16, 离名义的
#   CUTOFF=14:45 差了 30~45 分钟, 而且各只不一致。
#   实测这 45 分钟的信息差不可忽略: 14:00 与 14:45 两种截断出分的
#   Top100 重合只有 89%(对照"14:45 vs 全日"是 92%)。
#   14:25 再扫一遍, 每只刷新到 14:25~14:41, 离散度不变但整体近 25 分钟。
#   保留 14:00 那遍是**兜底** —— 二次预取失败时仍有当天数据可用。
#   两遍不能重叠: 每只 parquet 是读-改-写, 同时写会静默互相覆盖。
#   fetch_today_minutes 里有互斥锁(心跳判活), 撞上会显式失败而不是写坏。
#   根治仍是换批量取数(掘金 history() 可一次取多只) —— 逐只串行在
#   45 分钟窗口里拿不到同步的 14:45 截面, 推后触发只是平移区间。
#   14:50  place    新仿真账户收盘竞价下单
#   15:05  sync     回读成交
#   15:30  run      收盘后刷新日线 + 增量预测
#   19:30  reconcile 对账 (只读临时目录)
param(
    [switch]$List,
    [switch]$Apply,
    [switch]$Remove,
    [string]$PythonExe = "",
    [string]$WorkDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $WorkDir) { $WorkDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not ($WorkDir -match "csi-2000")) {
    Write-Host "工作目录必须是 csi-2000: $WorkDir" -ForegroundColor Red
    exit 2
}
if (-not $PythonExe) {
    $cands = @(
        "C:\Program Files\Python312\python.exe",
        "$env:USERPROFILE\.conda\envs\qlib\python.exe"
    )
    $PythonExe = $cands | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    Write-Host "找不到 Python 3.12" -ForegroundColor Red
    exit 2
}

$TASKS = @(
    @{ Name = "CSI2000-Prefetch"
       Args = "-X utf8 -m daily prefetch"
       Desc = "工作日 14:00 预取当日分钟(打底, 二次预取的兜底)"
       At = "14:00" },
    @{ Name = "CSI2000-Prefetch2"
       Args = "-X utf8 -m daily prefetch"
       Desc = "工作日 14:25 二次预取, 把截断时点推近 14:45(14:00 那遍兜底)"
       At = "14:25" },
    @{ Name = "CSI2000-Auction"
       Args = "-X utf8 -m daily auction"
       Desc = "工作日 14:46 截断出分"
       At = "14:46" },
    @{ Name = "CSI2000-Place"
       Args = "-X utf8 gm_bridge.py --place"
       Desc = "工作日 14:50 收盘竞价下单"
       At = "14:50" },
    @{ Name = "CSI2000-Sync"
       Args = "-X utf8 gm_bridge.py --sync"
       Desc = "工作日 15:05 回读成交"
       At = "15:05" },
    @{ Name = "CSI2000-Reconcile"
       Args = "-X utf8 reconcile.py --start 2024-01-02 --end 2027-12-31"
       Desc = "每日 19:30 全区间对账(短窗会漏掉挂单积压类问题)"
       At = "19:30" }
)

Write-Host "解释器: $PythonExe"
Write-Host "工作目录: $WorkDir"
if ($List -or (-not $Apply -and -not $Remove)) {
    $TASKS | ForEach-Object { Write-Host ("  {0}  {1}  {2}" -f $_.Name, $_.At, $_.Desc) }
    if (-not $Apply) {
        Write-Host "未注册。确认后加 -Apply"
        exit 0
    }
}

if ($Remove) {
    foreach ($t in $TASKS) {
        Unregister-ScheduledTask -TaskName $t.Name -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "已删除 $($t.Name)"
    }
    exit 0
}

foreach ($t in $TASKS) {
    $action = New-ScheduledTaskAction -Execute $PythonExe -Argument $t.Args -WorkingDirectory $WorkDir
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $t.At
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger -Settings $settings -Description $t.Desc -Force | Out-Null
    Write-Host "已注册 $($t.Name)"
}
