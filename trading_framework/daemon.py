#!/usr/bin/env python3
"""
飞书机器人守护进程
- 监控 smart_bot.py 是否运行
- 挂掉自动重启
- 记录日志
"""

import subprocess
import time
import os
import signal
import sys
from pathlib import Path
from datetime import datetime

# Windows 控制台默认非 UTF-8 编码（如 GBK），print 中文/emoji 会 UnicodeEncodeError 崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

WORK_DIR = Path(__file__).parent
BOT_SCRIPT = WORK_DIR / "smart_bot.py"
LOG_DIR = WORK_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "daemon.log"
PID_FILE = WORK_DIR / "daemon.pid"
CHECK_INTERVAL = 10  # 检查间隔（秒）
MAX_RESTARTS = 10    # 最大连续重启次数
RESTART_WINDOW = 300 # 重启计数窗口（秒）
CRASH_ALERT_THRESHOLD = 3  # 短时间内崩溃次数超此值则飞书告警
CRASH_ALERT_WINDOW = 600   # 告警统计窗口（秒，10分钟）

class Daemon:
    def __init__(self):
        self.bot_process = None
        self.restart_times = []
        self.running = True
        self._crash_alerted = False  # 避免重复告警

    def log(self, msg: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

    def start_bot(self):
        """启动机器人"""
        self.log("启动机器人...")
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        # Windows 中文环境的 locale 编码是 cp936。子进程若按 locale 读写文本，
        # 与用 -X utf8 启动的进程 (CLAUDE.md 里记录的命令都带该参数) 写出的文件
        # 会互相读不了。代码里已逐处显式指定 encoding='utf-8'，这里再兜一层：
        # 让子进程整体运行在 UTF-8 模式，避免遗漏处退化成 GBK。
        env['PYTHONUTF8'] = '1'
        # 保持日志文件句柄引用，避免泄漏；旧句柄在重启时关闭
        if hasattr(self, '_log_fh') and self._log_fh:
            try:
                self._log_fh.close()
            except OSError:
                pass
        self._log_fh = open(WORK_DIR / "logs" / "smart_bot.log", 'a',
                            encoding='utf-8', errors='replace')
        self.bot_process = subprocess.Popen(
            [sys.executable, '-u', str(BOT_SCRIPT)],
            cwd=str(WORK_DIR),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env
        )
        self.log(f"机器人已启动，PID: {self.bot_process.pid}")

    def check_bot(self) -> bool:
        """检查机器人是否运行"""
        if self.bot_process is None:
            return False
        return self.bot_process.poll() is None

    def _send_crash_alert(self, count: int, window: int):
        """进程频繁崩溃时发送飞书告警"""
        try:
            from send_signal import send_text
            msg = (
                f"🔴 Smart Bot 守护进程告警\n\n"
                f"Bot 进程在 {window // 60} 分钟内崩溃重启 {count} 次，"
                f"可能存在网络异常或程序错误。\n"
                f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            send_text(msg)
            self.log(f"[告警] 飞书崩溃告警已发送 (崩溃 {count} 次)")
        except Exception as e:
            self.log(f"[告警] 飞书推送失败: {e}")

    def restart_bot(self):
        """重启机器人"""
        now = time.time()
        # 清理过期的重启记录
        self.restart_times = [t for t in self.restart_times if now - t < RESTART_WINDOW]

        # 飞书告警：短时间内频繁崩溃
        recent_crashes = [t for t in self.restart_times if now - t < CRASH_ALERT_WINDOW]
        if len(recent_crashes) >= CRASH_ALERT_THRESHOLD and not self._crash_alerted:
            self._crash_alerted = True
            self._send_crash_alert(len(recent_crashes), CRASH_ALERT_WINDOW)

        # 检查是否重启过于频繁
        if len(self.restart_times) >= MAX_RESTARTS:
            self.log(f"⚠️ {RESTART_WINDOW}秒内重启{MAX_RESTARTS}次，暂停60秒...")
            time.sleep(60)
            self.restart_times.clear()
            self._crash_alerted = False  # 暂停后重置告警状态

        self.restart_times.append(now)

        # 确保旧进程已停止
        if self.bot_process:
            try:
                self.bot_process.kill()
                self.bot_process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.log(f"⚠ 清理旧进程失败 (PID: {self.bot_process.pid})")

        self.start_bot()

    def signal_handler(self, signum, frame):
        """处理退出信号"""
        self.log("收到退出信号，正在停止...")
        self.running = False
        if self.bot_process:
            self.bot_process.terminate()
            try:
                self.bot_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.bot_process.kill()
        if PID_FILE.exists():
            PID_FILE.unlink()
        sys.exit(0)

    def run(self):
        """主循环"""
        # 写入 PID
        with open(PID_FILE, 'w', encoding='utf-8') as f:
            f.write(str(os.getpid()))

        # 注册信号处理
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        self.log("=" * 50)
        self.log("守护进程启动")
        self.log(f"监控目标: {BOT_SCRIPT}")
        self.log(f"检查间隔: {CHECK_INTERVAL}秒")
        self.log("=" * 50)

        # 首次启动
        self.start_bot()

        # 监控循环
        while self.running:
            time.sleep(CHECK_INTERVAL)

            if not self.check_bot():
                self.log("❌ 机器人已停止，正在重启...")
                self.restart_bot()
            else:
                # 每分钟输出一次心跳
                if int(time.time()) % 60 < CHECK_INTERVAL:
                    self.log("💓 机器人运行正常")


IS_WINDOWS = os.name == 'nt'


def pid_alive(pid: int) -> bool:
    """检查进程是否存活

    不能用 os.kill(pid, 0)。Windows 上 os.kill 对非 CTRL_* 信号走的是
    TerminateProcess —— 传 0 不是探测而是**真的杀掉进程**(退出码 0)，
    进程已不存在时又抛 SystemError 包裹的 WinError 87 而非 ProcessLookupError。
    原实现因此既会误杀正在运行的守护进程，也接不住异常。
    """
    if IS_WINDOWS:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True, errors='replace'
        )
        return str(pid) in (result.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    # 检查是否已有守护进程运行
    if PID_FILE.exists():
        old_pid = int(PID_FILE.read_text().strip())
        if pid_alive(old_pid):
            print(f"守护进程已在运行 (PID: {old_pid})")
            print("如需重启，请先运行: python daemon.py stop")
            sys.exit(1)
        PID_FILE.unlink()  # 清理残留的 PID 文件

    daemon = Daemon()
    daemon.run()


def stop():
    """停止守护进程"""
    if not PID_FILE.exists():
        print("守护进程未运行")
        return

    pid = int(PID_FILE.read_text().strip())
    if not pid_alive(pid):
        print("守护进程已不存在")
        PID_FILE.unlink(missing_ok=True)
        return

    if IS_WINDOWS:
        # Windows 上 SIGTERM 即 TerminateProcess，signal_handler 不会执行，
        # 子进程 smart_bot 会被遗留成孤儿并继续占用飞书 WebSocket 连接
        # (导致重启后出现两个 bot 重复处理消息)。用 /T 连同子进程树一起终止。
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                       capture_output=True, text=True, errors='replace')
        print(f"已终止守护进程及其子进程 (PID: {pid})")
    else:
        os.kill(pid, signal.SIGTERM)
        print(f"已发送停止信号到守护进程 (PID: {pid})")

    for _ in range(10):
        if not pid_alive(pid):
            print("守护进程已停止")
            break
        time.sleep(0.5)
    else:
        print(f"⚠ 守护进程 {pid} 仍在运行，请手动检查")
        return

    PID_FILE.unlink(missing_ok=True)


def status():
    """查看状态"""
    if not PID_FILE.exists():
        print("守护进程未运行")
        return

    pid = int(PID_FILE.read_text().strip())
    if not pid_alive(pid):
        print("守护进程已停止（PID文件残留）")
        return

    print(f"守护进程运行中 (PID: {pid})")
    print("\n最近日志:")
    try:
        lines = LOG_FILE.read_text(encoding='utf-8', errors='replace').splitlines()
        for line in lines[-10:]:
            print(line)
    except OSError as e:
        print(f"(读取日志失败: {e})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "stop":
            stop()
        elif cmd == "status":
            status()
        elif cmd == "restart":
            stop()
            time.sleep(2)
            main()
        else:
            print("用法: python daemon.py [start|stop|status|restart]")
    else:
        main()
