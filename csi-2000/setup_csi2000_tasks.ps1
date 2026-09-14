# CSI2000 专用定时任务。名称带 CSI2000- 前缀, 不会覆盖 CSI300 的 TradingSystem-*。
#
# 默认只列出计划, 不注册。确认机器上没有 CSI300 同名冲突后再:
#     powershell -ExecutionPolicy Bypass -File setup_csi2000_tasks.ps1 -Apply
#
# 时钟:
#   14:00  prefetch 当日分钟 (给 14:45 出分)
#   14:46  auction  路线 B 截断出分 (模型未落盘会失败退出, 不误下单)
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
       Desc = "工作日 14:00 预取当日分钟"
       At = "14:00" },
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
