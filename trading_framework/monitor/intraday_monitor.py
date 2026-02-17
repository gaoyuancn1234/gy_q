#!/usr/bin/env python3
"""盘中监控 — 止损 + 异动 + 订单提醒

launchd 每天 9:25 启动，15:05 自动退出。
内部 5 分钟轮询 Sina 实时行情。

用法:
    python monitor/intraday_monitor.py              # 正常运行 (循环)
    python monitor/intraday_monitor.py --dry-run    # 单次检查 (不推送, 不循环)
    python monitor/intraday_monitor.py --once       # 单次检查 + 推送
"""
import os
import sys
import json
import re
import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env", override=True)

from config.settings import STOP_LOSS

# 日志
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "intraday_monitor.log", encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# 常量
HOLDINGS_FILE = PROJECT_DIR / "portfolio" / "live_holdings.json"
STATE_FILE = PROJECT_DIR / "monitor" / "monitor_state.json"
CHECK_INTERVAL = 300  # 5 分钟
INTRADAY_DROP_THRESHOLD = -0.05  # 日内跌 5%
PENDING_REMINDER_TIMES = ["09:35", "13:05"]  # 待执行订单提醒时间


# ============ Sina 实时行情 ============

def _qlib_to_sina(code: str) -> str:
    """SH600036 → sh600036"""
    return code[:2].lower() + code[2:]


def fetch_realtime_prices(instruments: list[str]) -> dict:
    """Sina API 批量获取实时价格

    Returns:
        {'SH600036': {'price': 42.3, 'open': 42.1, 'pre_close': 42.05,
                      'high': 42.5, 'low': 41.8, 'name': '招商银行'}, ...}
    """
    if not instruments:
        return {}

    # 批量请求，一次最多 50 只
    results = {}
    for i in range(0, len(instruments), 50):
        batch = instruments[i:i+50]
        sina_codes = [_qlib_to_sina(c) for c in batch]
        url = f"http://hq.sinajs.cn/list={','.join(sina_codes)}"

        req = Request(url)
        req.add_header('Referer', 'http://finance.sina.com.cn')

        raw = None
        for attempt in range(2):
            try:
                with urlopen(req, timeout=10) as resp:
                    raw = resp.read().decode('gbk')
                break
            except (URLError, OSError) as e:
                if attempt == 0:
                    log.warning(f"Sina API 请求失败 (重试中): {e}")
                    time.sleep(2)
                else:
                    log.error(f"Sina API 请求失败 (已重试): {e}")
        if raw is None:
            continue

        for line in raw.strip().split('\n'):
            line = line.strip()
            if not line or '=' not in line:
                continue
            # var hq_str_sh600036="招商银行,42.10,42.05,42.30,..."
            match = re.match(r'var hq_str_(\w+)="(.*)";', line)
            if not match:
                continue
            sina_code = match.group(1)
            fields = match.group(2).split(',')
            if len(fields) < 9 or not fields[3]:
                continue

            # 还原 Qlib code
            qlib_code = sina_code[:2].upper() + sina_code[2:]
            try:
                price = float(fields[3])
                if price <= 0:
                    continue
                results[qlib_code] = {
                    'name': fields[0],
                    'open': float(fields[1]) if fields[1] else 0,
                    'pre_close': float(fields[2]) if fields[2] else 0,
                    'price': price,
                    'high': float(fields[4]) if fields[4] else 0,
                    'low': float(fields[5]) if fields[5] else 0,
                }
            except (ValueError, IndexError):
                continue

    return results


# ============ 告警检查 ============

def check_alerts(holdings: dict, prices: dict, alerted: dict) -> list[dict]:
    """检查所有告警条件，返回新触发的告警列表"""
    alerts = []
    now = datetime.now()
    time_str = now.strftime('%H:%M')

    positions = holdings.get('positions', {})

    for code, pos in positions.items():
        p = prices.get(code)
        if not p:
            continue

        cost = pos.get('cost_price', 0)
        current = p['price']
        name = p.get('name', pos.get('name', code))

        # 1. 止损: 当前价 vs 成本价
        if cost > 0:
            loss_pct = (current - cost) / cost
            key = f"{code}:stop_loss"
            if loss_pct <= -STOP_LOSS and key not in alerted:
                alerts.append({
                    'type': 'stop_loss',
                    'code': code,
                    'name': name,
                    'current': current,
                    'cost': cost,
                    'pct': loss_pct,
                })
                alerted[key] = True

        # 2. 日内急跌: 当前价 vs 今开
        open_price = p.get('open', 0)
        if open_price > 0:
            intraday_pct = (current - open_price) / open_price
            key = f"{code}:intraday_drop"
            if intraday_pct <= INTRADAY_DROP_THRESHOLD and key not in alerted:
                alerts.append({
                    'type': 'intraday_drop',
                    'code': code,
                    'name': name,
                    'current': current,
                    'open': open_price,
                    'pct': intraday_pct,
                })
                alerted[key] = True

    # 3. 待执行订单提醒 (9:30~9:39 → morning, 13:00~13:09 → afternoon)
    pending = holdings.get('pending_orders', {})
    has_pending = pending.get('sells') or pending.get('buys')
    hhmm = now.hour * 100 + now.minute
    if has_pending and (930 <= hhmm <= 939 or 1300 <= hhmm <= 1309):
        key = f"pending:{'morning' if hhmm < 1200 else 'afternoon'}"
        if key not in alerted:
            sell_count = len(pending.get('sells', []))
            buy_count = len(pending.get('buys', {}))
            alerts.append({
                'type': 'pending_reminder',
                'sells': sell_count,
                'buys': buy_count,
            })
            alerted[key] = True

    return alerts


# ============ 消息格式化 ============

def format_alert_message(alerts: list[dict]) -> str:
    """格式化告警消息"""
    now_str = datetime.now().strftime('%H:%M')
    lines = [f"🚨 盘中预警 ({now_str})", ""]

    stop_loss_alerts = [a for a in alerts if a['type'] == 'stop_loss']
    drop_alerts = [a for a in alerts if a['type'] == 'intraday_drop']
    pending_alerts = [a for a in alerts if a['type'] == 'pending_reminder']

    if stop_loss_alerts:
        lines.append("【止损触发】")
        for a in stop_loss_alerts:
            num = a['code'][2:]
            lines.append(
                f"  {a['name']}({num}) 当前 {a['current']:.2f} | "
                f"成本 {a['cost']:.2f} | 亏损 {a['pct']:.1%}"
            )
            lines.append("  ⚡ 建议立即卖出")
        lines.append("")

    if drop_alerts:
        lines.append("【日内急跌】")
        for a in drop_alerts:
            num = a['code'][2:]
            lines.append(
                f"  {a['name']}({num}) 开盘 {a['open']:.2f} → "
                f"当前 {a['current']:.2f} | 跌 {a['pct']:.1%}"
            )
        lines.append("")

    if pending_alerts:
        lines.append("【待执行订单】")
        for a in pending_alerts:
            lines.append(f"  卖出 {a['sells']} 只 | 买入 {a['buys']} 只")
            lines.append("  请尽快执行调仓指令")
        lines.append("")

    return "\n".join(lines)


def _get_model_version() -> str:
    """读取模型版本号"""
    config_file = PROJECT_DIR / "config" / "signal_config.yaml"
    try:
        import yaml
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        tag = cfg.get('model_tag', 'M01')
        retrain = cfg.get('last_retrain', '')  # '2026-02'
        if retrain:
            parts = retrain.split('-')
            ver = parts[0][2:] + parts[1] if len(parts) == 2 else retrain.replace('-', '')
            return f"{tag}-v{ver}"
        return tag
    except Exception:
        return "M01"


def format_closing_summary(holdings: dict, prices: dict) -> str:
    """15:00 收盘总结"""
    positions = holdings.get('positions', {})
    cash = holdings.get('cash', 0)
    initial = holdings.get('initial_capital', 100000)

    total_market = 0
    stock_lines = []
    for code, pos in positions.items():
        p = prices.get(code)
        if not p:
            continue
        current = p['price']
        shares = pos.get('shares', 0)
        cost = pos.get('cost_price', 0)
        market = shares * current
        total_market += market

        pnl = (current - cost) / cost if cost > 0 else 0
        # 日内涨跌
        pre_close = p.get('pre_close', 0)
        day_pnl = (current - pre_close) / pre_close if pre_close > 0 else 0

        sign_day = "+" if day_pnl >= 0 else ""
        sign_total = "+" if pnl >= 0 else ""
        name = p.get('name', pos.get('name', code))
        num = code[2:]
        stock_lines.append(
            f"  {name}({num}) {current:.2f} "
            f"今日{sign_day}{day_pnl:.1%} | 总{sign_total}{pnl:.1%}"
        )

    total = total_market + cash
    total_pnl = (total - initial) / initial
    sign = "+" if total_pnl >= 0 else ""

    model_ver = _get_model_version()
    lines = [
        f"📊 收盘总结 ({datetime.now().strftime('%H:%M')})",
        f"模型: {model_ver}",
        "",
        f"总资产: {total:,.0f} ({sign}{total_pnl:.1%})",
    ]
    if stock_lines:
        lines.append("")
        lines.extend(stock_lines)
    else:
        lines.append("当前空仓")

    lines.append(f"\n现金: {cash:,.0f}")

    return "\n".join(lines)


# ============ 飞书推送 ============

def push_feishu(message: str):
    """推送消息到飞书 (复用 daily_runner 模式)"""
    try:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        app_id = os.environ.get("FEISHU_APP_ID_1", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET_1", "")
        user_id = os.environ.get("FEISHU_USER_OPEN_ID", "")

        if not all([app_id, app_secret, user_id]):
            log.error("飞书凭证未配置")
            log.info(f"消息内容:\n{message}")
            return

        client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .build()

        req = CreateMessageRequest.builder() \
            .receive_id_type("open_id") \
            .request_body(CreateMessageRequestBody.builder()
                .receive_id(user_id)
                .msg_type("text")
                .content(json.dumps({"text": message}))
                .build()) \
            .build()

        resp = client.im.v1.message.create(req)
        if resp.success():
            log.info("飞书推送成功")
        else:
            log.error(f"飞书推送失败: {resp.code} - {resp.msg}")
    except Exception as e:
        log.error(f"飞书推送异常: {e}")
        log.info(f"消息内容:\n{message}")


# ============ 状态持久化 ============

def load_monitor_state() -> dict:
    """加载监控状态，新的一天自动重置"""
    today = date.today().isoformat()
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            if state.get('date') == today:
                return state
        except (json.JSONDecodeError, KeyError):
            pass

    return {
        'date': today,
        'alerted_today': {},
        'check_count': 0,
        'last_check': None,
        'started_at': datetime.now().strftime('%H:%M:%S'),
    }


def save_monitor_state(state: dict):
    """保存监控状态 (原子写入防 kill 导致损坏)"""
    tmp = STATE_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


# ============ 交易日检查 ============

def is_trading_day() -> bool:
    """复用 daily_runner 的交易日逻辑"""
    today = date.today()
    if today.weekday() >= 5:
        return False
    try:
        import baostock as bs
        lg = bs.login()
        try:
            rs = bs.query_trade_dates(
                start_date=today.strftime('%Y-%m-%d'),
                end_date=today.strftime('%Y-%m-%d'),
            )
            while rs.next():
                row = rs.get_row_data()
                if row[1] == '1':
                    return True
            return False
        finally:
            bs.logout()
    except Exception as e:
        log.warning(f"交易日检查失败: {e}, 按工作日处理")
        return today.weekday() < 5


# ============ 主循环 ============

def _check_virtual_portfolios() -> str:
    """检查是否有活跃的 shadow 或 experiment 虚拟盘"""
    parts = []
    try:
        from shadow_manager import ShadowManager
        sm = ShadowManager()
        active = sm.get_active_candidates()
        if active:
            parts.append(f"影子{len(active)}个")
    except Exception:
        pass
    try:
        from experiment_manager import ExperimentManager
        em = ExperimentManager()
        active = em.get_active_experiments()
        if active:
            parts.append(f"实验{len(active)}个")
    except Exception:
        pass
    return ", ".join(parts)


def run_single_check(dry_run: bool = False) -> dict:
    """单次检查，返回结果摘要"""
    # 加载持仓
    if not HOLDINGS_FILE.exists():
        log.info("无持仓文件")
        return {'status': 'no_holdings'}

    with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
        holdings = json.load(f)

    positions = holdings.get('positions', {})
    instruments = list(positions.keys())

    # 无持仓时仅检查 pending_orders
    pending = holdings.get('pending_orders', {})
    has_pending = pending.get('sells') or pending.get('buys')

    if not instruments and not has_pending:
        # 检查是否有活跃的虚拟盘 (shadow/experiment)
        virtual_active = _check_virtual_portfolios()
        if virtual_active:
            log.info(f"实盘空仓，但虚拟盘活跃: {virtual_active}")
            return {'status': 'empty_but_virtual', 'virtual': virtual_active}
        log.info("空仓且无待执行订单")
        return {'status': 'empty'}

    # 获取实时价格
    prices = fetch_realtime_prices(instruments) if instruments else {}
    if instruments and not prices:
        log.warning("获取实时价格失败")
        return {'status': 'price_error'}

    # 加载状态
    state = load_monitor_state()
    alerted = state.get('alerted_today', {})

    # 检查告警
    alerts = check_alerts(holdings, prices, alerted)

    # 更新状态
    state['alerted_today'] = alerted
    state['check_count'] = state.get('check_count', 0) + 1
    state['last_check'] = datetime.now().strftime('%H:%M:%S')
    save_monitor_state(state)

    # 发送告警
    if alerts:
        message = format_alert_message(alerts)
        if dry_run:
            log.info(f"[DRY RUN] 告警消息:\n{message}")
        else:
            push_feishu(message)
        log.info(f"触发 {len(alerts)} 条告警")
    else:
        log.info(f"检查完成，无新告警 (持仓{len(positions)}只, 已触发{len(alerted)}条)")

    return {
        'status': 'ok',
        'positions': len(positions),
        'alerts': len(alerts),
        'prices': {k: v['price'] for k, v in prices.items()},
    }


def run_monitor_loop(dry_run: bool = False):
    """主循环: 9:25 启动 → 9:30 开始 → 15:05 退出"""
    log.info("=" * 40)
    log.info("盘中监控启动")
    log.info("=" * 40)

    # 等待 9:30 开盘
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now < market_open:
        wait = (market_open - now).total_seconds()
        log.info(f"等待开盘 ({wait:.0f}s)...")
        time.sleep(wait)

    sent_closing = False

    while True:
        now = datetime.now()
        hm = now.hour * 100 + now.minute  # 930, 1130, 1300, 1505

        # 15:05 后退出
        if hm >= 1505:
            if not sent_closing:
                _send_closing_summary(dry_run)
                sent_closing = True
            log.info("盘后，监控退出")
            break

        # 15:00 收盘总结
        if hm >= 1500 and not sent_closing:
            _send_closing_summary(dry_run)
            sent_closing = True

        # 11:30~13:00 午休跳过
        if 1130 <= hm < 1300:
            # 等到 13:00
            resume = now.replace(hour=13, minute=0, second=0, microsecond=0)
            wait = (resume - now).total_seconds()
            if wait > 0:
                log.info(f"午休，等待 {wait:.0f}s...")
                time.sleep(wait)
            continue

        # 执行检查
        result = run_single_check(dry_run)

        # 空仓处理
        status = result.get('status', '')
        if status in ('empty', 'no_holdings'):
            # 完全空仓 + 无虚拟盘: 低频 30 分钟
            log.info("空仓低频模式: 30分钟后再检查")
            idle_minutes = 30
        elif status == 'empty_but_virtual':
            # 实盘空仓但虚拟盘活跃: 中频 15 分钟
            log.info(f"实盘空仓，虚拟盘活跃 ({result.get('virtual', '')}): 15分钟后再检查")
            idle_minutes = 15
        else:
            idle_minutes = 0

        if idle_minutes > 0:
            next_check_time = now + timedelta(minutes=idle_minutes)
            close_time = now.replace(hour=15, minute=5, second=0, microsecond=0)
            if next_check_time > close_time:
                next_check_time = close_time
            wait = (next_check_time - datetime.now()).total_seconds()
            if wait > 0:
                time.sleep(wait)
            continue

        # 等到下一个 5 分钟整点
        now = datetime.now()
        next_min = (now.minute // 5 + 1) * 5
        if next_min >= 60:
            next_check = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_check = now.replace(minute=next_min, second=0, microsecond=0)
        wait = (next_check - now).total_seconds()
        if wait > 0:
            time.sleep(wait)


def _send_closing_summary(dry_run: bool):
    """发送收盘总结"""
    if not HOLDINGS_FILE.exists():
        return
    with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
        holdings = json.load(f)
    instruments = list(holdings.get('positions', {}).keys())
    if not instruments:
        return
    prices = fetch_realtime_prices(instruments)
    if not prices:
        return
    message = format_closing_summary(holdings, prices)
    if dry_run:
        log.info(f"[DRY RUN] 收盘总结:\n{message}")
    else:
        push_feishu(message)
    log.info("收盘总结已发送")


# ============ 入口 ============

def main():
    import argparse
    parser = argparse.ArgumentParser(description='盘中监控')
    parser.add_argument('--dry-run', action='store_true', help='单次检查，不推送不循环')
    parser.add_argument('--once', action='store_true', help='单次检查 + 推送')
    parser.add_argument('--force', action='store_true', help='忽略交易日检查')
    args = parser.parse_args()

    if not args.force and not args.dry_run:
        if not is_trading_day():
            log.info("今天非交易日，跳过")
            return

    if args.dry_run:
        log.info("--- DRY RUN 模式 ---")
        result = run_single_check(dry_run=True)
        log.info(f"结果: {result}")
    elif args.once:
        log.info("--- 单次检查模式 ---")
        run_single_check(dry_run=False)
    else:
        run_monitor_loop()


if __name__ == '__main__':
    main()
