#!/usr/bin/env python3
"""每日收盘后: 刷新数据 → 生成信号 → 推送飞书

Cron/launchd: 工作日 18:00

用法:
    python daily_runner.py              # 正常运行
    python daily_runner.py --dry-run    # 只生成不推送
    python daily_runner.py --force      # 忽略交易日检查
"""
import os
import sys
import json
import time
import logging
from datetime import datetime, date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(PROJECT_DIR / ".env", override=True)

# 日志
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "daily_runner.log", encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ============ 交易日检查 ============

def is_trading_day() -> bool:
    """检查今天是否为 A 股交易日 (周一~周五, 非节假日)"""
    today = date.today()
    # 周末直接跳过
    if today.weekday() >= 5:
        return False
    # 简单节假日检查: 用 BaoStock 日历判断
    try:
        import baostock as bs
        lg = bs.login()
        rs = bs.query_trade_dates(
            start_date=today.strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
        )
        while rs.next():
            row = rs.get_row_data()
            if row[1] == '1':  # is_trading_day
                bs.logout()
                return True
        bs.logout()
        return False
    except Exception as e:
        log.warning(f"交易日检查失败: {e}, 按工作日处理")
        return today.weekday() < 5


# ============ 数据刷新 ============

def refresh_daily_data() -> bool:
    """增量刷新 BaoStock 数据"""
    log.info("刷新 BaoStock 数据...")
    try:
        from qlib_engine.data_setup import setup_qlib_data
        t0 = time.time()
        setup_qlib_data()
        log.info(f"数据刷新完成 ({time.time() - t0:.0f}s)")
        return True
    except Exception as e:
        log.error(f"数据刷新失败: {e}")
        return False


# ============ 信号生成 + 推送 ============

def generate_and_push(dry_run: bool = False):
    """生成信号 → 推送飞书"""
    from factor_lab.signal_generator import SignalGenerator
    from portfolio.live_portfolio import (
        load_live_holdings, save_live_holdings,
        get_current_prices, generate_live_instructions,
        check_stop_loss, get_daily_report, get_stock_names,
    )

    log.info("加载信号生成器...")
    sg = SignalGenerator()

    # 获取最新信号
    signal = sg.get_signal()
    if 'error' in signal:
        log.error(f"信号生成失败: {signal['error']}")
        push_feishu(f"❌ ML信号生成失败: {signal['error']}", dry_run)
        return

    log.info(f"信号日期: {signal['date']}, 状态: {signal['regime']}, TopK: {signal['effective_topk']}")

    # 信号时效性检查: 超过2个交易日标记为 stale
    signal_stale = False
    try:
        signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
        days_gap = (date.today() - signal_date).days
        if days_gap > 3:  # 日历日>3 约等于交易日>2
            signal_stale = True
            log.warning(f"信号已过期: 信号日期 {signal['date']}, 距今 {days_gap} 天")
    except (ValueError, KeyError):
        pass

    # 加载持仓
    holdings = load_live_holdings()

    # 更新调仓计数
    holdings['rebalance_day_count'] = holdings.get('rebalance_day_count', 0) + 1

    # 获取价格 (当前持仓 + 目标股票)
    all_instruments = list(set(
        list(holdings.get('positions', {}).keys()) + signal['target_stocks']
    ))
    prices = get_current_prices(all_instruments)
    if not prices:
        log.error("获取价格失败")
        push_feishu("❌ 获取实时价格失败，请检查 BaoStock 连接", dry_run)
        return

    # 判断是否为调仓日 (每 5 个交易日)
    rebalance_every = 5
    is_rebalance = (holdings['rebalance_day_count'] % rebalance_every == 0)
    log.info(f"交易日计数: {holdings['rebalance_day_count']}, 调仓: {'是' if is_rebalance else '否'}")

    if is_rebalance:
        # ── 调仓日: 生成调仓指令 ──
        message = generate_live_instructions(signal, holdings, prices)

        # 保存待执行订单到 holdings
        target_set = set(signal['target_stocks'])
        current_set = set(holdings.get('positions', {}).keys())
        to_sell = list(current_set - target_set)
        to_buy = [c for c in signal['target_stocks'] if c not in current_set]

        holdings['pending_orders'] = {
            'sells': to_sell,
            'buys': {c: signal['scores'].get(c, 0) for c in to_buy},
        }
        holdings['last_signal_date'] = signal['date']
        save_live_holdings(holdings)

        log.info(f"调仓指令: 卖{len(to_sell)} 买{len(to_buy)}")
    else:
        # ── 非调仓日: 日报 ──
        message = get_daily_report(holdings, prices)

        # 止损检查
        alerts = check_stop_loss(holdings, prices)
        if alerts:
            alert_names = [f"{a['name']}({a['loss_pct']:.1%})" for a in alerts]
            message += f"\n\n🚨 止损触发: {', '.join(alert_names)}"
            message += "\n建议立即卖出上述股票"
            log.warning(f"止损预警: {alert_names}")

        save_live_holdings(holdings)

    # 信号过期提示
    if signal_stale:
        days_gap = (date.today() - signal_date).days
        message += f"\n\n⚠️ 信号已过期: 信号日期 {signal['date']}，距今 {days_gap} 天，请关注数据刷新是否正常"

    # 检查模型新鲜度
    freshness = sg.check_model_freshness()
    if freshness['is_stale']:
        message += f"\n\n⚠️ {freshness['message']}\n发送「重训」执行季度重训"

    push_feishu(message, dry_run)


# ============ 飞书推送 ============

def push_feishu(message: str, dry_run: bool = False):
    """推送消息到飞书"""
    if dry_run:
        log.info(f"[DRY RUN] 飞书消息:\n{message}")
        return

    try:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        app_id = os.environ.get("FEISHU_APP_ID_1", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET_1", "")
        user_id = os.environ.get("FEISHU_USER_OPEN_ID", "")

        if not all([app_id, app_secret, user_id]):
            log.error("飞书凭证未配置，消息未发送")
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
            log.info("飞书消息发送成功")
        else:
            log.error(f"飞书发送失败: {resp.code} - {resp.msg}")
    except Exception as e:
        log.error(f"飞书推送异常: {e}")
        log.info(f"消息内容:\n{message}")


# ============ 主入口 ============

def main():
    import argparse
    parser = argparse.ArgumentParser(description='每日信号推送')
    parser.add_argument('--dry-run', action='store_true', help='只生成不推送')
    parser.add_argument('--force', action='store_true', help='忽略交易日检查')
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("Daily Runner 启动")
    log.info("=" * 50)

    # 交易日检查
    if not args.force and not is_trading_day():
        log.info("今天非交易日，跳过")
        return

    # 刷新数据 (失败则用旧数据继续，但记录警告)
    if not refresh_daily_data():
        log.warning("数据刷新失败，将使用缓存数据继续")

    # 生成信号 + 推送
    generate_and_push(dry_run=args.dry_run)

    log.info("Daily Runner 完成")


if __name__ == '__main__':
    main()
