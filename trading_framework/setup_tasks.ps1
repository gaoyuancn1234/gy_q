# 重建全部 Windows 定时任务
#
# 这些任务此前是逐个手工创建的，换机器就得从头再来一遍，而且极易漏配
# WorkingDirectory —— 2026-09-04 发现原先该项为空，相对路径全解析到
# System32，任务显示"成功"实则什么都没干。这里显式写死每一项。
#
# 用法 (在 trading_framework 目录下):
#     powershell -ExecutionPolicy Bypass -File setup_tasks.ps1
#     powershell -ExecutionPolicy Bypass -File setup_tasks.ps1 -List
#     powershell -ExecutionPolicy Bypass -File setup_tasks.ps1 -Remove
#
# 任务默认"仅在用户登录时运行"。若需未登录也跑，得在任务计划程序 GUI 里
# 改为"不管用户是否登录都运行"并存储账户密码 —— 脚本无法代劳。

param(
    [switch]$List,
    [switch]$Remove,
    [string]$PythonExe = "",
    [string]$WorkDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $WorkDir) { $WorkDir = (Get-Location).Path }
if (-not $PythonExe) {
    # 解释器因机器而异，必须实测存在。系统自带的 3.13 装不了 pyqlib。
    $cands = @(
        "$env:USERPROFILE\.conda\envs\qlib\python.exe",
        "C:\Program Files\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    )
    $PythonExe = $cands | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    Write-Host "✗ 找不到可用的 Python。用 -PythonExe 显式指定:" -ForegroundColor Red
    Write-Host "    .\setup_tasks.ps1 -PythonExe C:\path\to\python.exe"
    exit 2
}
if (-not (Test-Path (Join-Path $WorkDir "daily_runner.py"))) {
    Write-Host "✗ $WorkDir 里没有 daily_runner.py —— 请在 trading_framework 目录下运行" -ForegroundColor Red
    exit 2
}

Write-Host "解释器: $PythonExe"
Write-Host "工作目录: $WorkDir"
Write-Host ""

# 名称, 参数, 计划描述, 触发器构造
$TASKS = @(
    @{ Name = "TradingSystem-DailyRunner"
       Args = "-X utf8 daily_runner.py"
       Desc = "每日 18:00 信号推送"
       Trigger = { New-ScheduledTaskTrigger -Daily -At 18:00 } },

    @{ Name = "TradingSystem-PaperTrader"
       Args = "-X utf8 -m factor_lab.paper_trader replay"
       Desc = "工作日 18:30 模拟盘重放"
       Trigger = { New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 18:30 } },

    @{ Name = "TradingSystem-Reconcile"
       Args = "-X utf8 reconcile.py --days 120 --push"
       Desc = "每日 19:30 双路径对账，发现分叉推飞书"
       Trigger = { New-ScheduledTaskTrigger -Daily -At 19:30 } },

    @{ Name = "TradingSystem-FactorMiner"
       Args = "-X utf8 -m factor_lab.factor_miner --daily"
       Desc = "每周二/五 22:00 因子挖掘"
       Trigger = { New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday,Friday -At 22:00 } },

    @{ Name = "TradingSystem-SelfReflect"
       Args = "-X utf8 self_reflect.py"
       Desc = "每日 23:30 自我反思"
       Trigger = { New-ScheduledTaskTrigger -Daily -At 23:30 } },

    @{ Name = "TradingSystem-FeishuBot"
       Args = "-X utf8 daemon.py"
       Desc = "飞书机器人守护进程 (登录后自启)"
       Trigger = { New-ScheduledTaskTrigger -AtLogOn } }
)

if ($List) {
    Get-ScheduledTask -TaskName "TradingSystem-*" -ErrorAction SilentlyContinue |
        ForEach-Object {
            $i = Get-ScheduledTaskInfo -TaskName $_.TaskName
            "{0,-32} {1,-8} 上次 {2} 结果 {3}" -f $_.TaskName, $_.State,
                $i.LastRunTime, $i.LastTaskResult
        }
    exit 0
}

if ($Remove) {
    Get-ScheduledTask -TaskName "TradingSystem-*" -ErrorAction SilentlyContinue |
        ForEach-Object {
            Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false
            Write-Host "已删除 $($_.TaskName)"
        }
    exit 0
}

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$ok = 0
$fail = @()

foreach ($t in $TASKS) {
    try {
        $act = New-ScheduledTaskAction -Execute $PythonExe -Argument $t.Args -WorkingDirectory $WorkDir
        $trg = & $t.Trigger
        $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 6)
        Register-ScheduledTask -TaskName $t.Name -Action $act -Trigger $trg `
            -Settings $set -Principal $principal -Description $t.Desc -Force | Out-Null
        # 注册完必须回查 —— 不拿"没抛异常"当创建成功
        $chk = Get-ScheduledTask -TaskName $t.Name -ErrorAction SilentlyContinue
        if ($chk) {
            Write-Host ("✓ {0,-32} {1}" -f $t.Name, $t.Desc) -ForegroundColor Green
            $ok++
        } else {
            Write-Host ("✗ {0,-32} 注册后回查不到" -f $t.Name) -ForegroundColor Red
            $fail += $t.Name
        }
    } catch {
        Write-Host ("✗ {0,-32} {1}" -f $t.Name, $_.Exception.Message) -ForegroundColor Red
        $fail += $t.Name
    }
}

# 盘中监控要每 5 分钟重复，New-ScheduledTaskTrigger 不直接支持，单独处理
try {
    $act = New-ScheduledTaskAction -Execute $PythonExe `
        -Argument "-X utf8 monitor/intraday_monitor.py --once" -WorkingDirectory $WorkDir
    $trg = New-ScheduledTaskTrigger -Daily -At 9:25
    $trg.Repetition = (New-ScheduledTaskTrigger -Once -At 9:25 `
        -RepetitionInterval (New-TimeSpan -Minutes 5) `
        -RepetitionDuration (New-TimeSpan -Hours 5 -Minutes 40)).Repetition
    $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    Register-ScheduledTask -TaskName "TradingSystem-IntradayMonitor" -Action $act `
        -Trigger $trg -Settings $set -Principal $principal `
        -Description "9:25~15:05 每 5 分钟盘中监控" -Force | Out-Null
    if (Get-ScheduledTask -TaskName "TradingSystem-IntradayMonitor" -ErrorAction SilentlyContinue) {
        Write-Host ("✓ {0,-32} {1}" -f "TradingSystem-IntradayMonitor", "9:25~15:05 每 5 分钟") -ForegroundColor Green
        $ok++
    } else { $fail += "TradingSystem-IntradayMonitor" }
} catch {
    Write-Host ("✗ TradingSystem-IntradayMonitor  {0}" -f $_.Exception.Message) -ForegroundColor Red
    $fail += "TradingSystem-IntradayMonitor"
}

Write-Host ""
if ($fail.Count -eq 0) {
    Write-Host "全部 $ok 个任务创建成功" -ForegroundColor Green
} else {
    Write-Host "$ok 个成功，$($fail.Count) 个失败: $($fail -join ', ')" -ForegroundColor Red
    exit 1
}
