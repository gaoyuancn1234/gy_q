#!/usr/bin/env python3
"""
发送调仓信号到飞书
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
import lark_oapi as lark
from lark_oapi.api.im.v1 import *

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

APP_ID = os.environ.get("FEISHU_APP_ID_1", "")
APP_SECRET = os.environ.get("FEISHU_APP_SECRET_1", "")
USER_OPEN_ID = os.environ.get("FEISHU_USER_OPEN_ID", "")
CHAT_ID = ""


def get_client():
    """获取飞书客户端"""
    return lark.Client.builder() \
        .app_id(APP_ID) \
        .app_secret(APP_SECRET) \
        .build()


def send_text(text: str, user_id: str = None, chat_id: str = None):
    """发送文本消息"""
    client = get_client()

    target_id = chat_id or CHAT_ID or user_id or USER_OPEN_ID
    id_type = "chat_id" if (chat_id or CHAT_ID) else "open_id"

    if not target_id:
        print("错误: 请配置 USER_OPEN_ID 或 CHAT_ID")
        print("\n获取方式:")
        print("1. 在飞书中私聊机器人，机器人会收到你的 open_id")
        print("2. 或将机器人拉入群聊，获取 chat_id")
        return False

    request = CreateMessageRequest.builder() \
        .receive_id_type(id_type) \
        .request_body(CreateMessageRequestBody.builder()
            .receive_id(target_id)
            .msg_type("text")
            .content(json.dumps({"text": text}))
            .build()) \
        .build()

    response = client.im.v1.message.create(request)

    if response.success():
        print("✓ 消息发送成功")
        return True
    else:
        print(f"✗ 发送失败: {response.code} - {response.msg}")
        return False


def generate_signal_message():
    """生成调仓信号消息"""
    from portfolio.trade_executor import (
        load_holdings, get_current_prices, get_top12_signals,
        STOCK_NAMES
    )

    holdings = load_holdings()

    if holdings['total_capital'] is None:
        return "❌ 请先设置总资金"

    print("获取市场数据...")
    result = get_current_prices()
    if result[0] is None:
        return "❌ 无法获取市场数据"

    prices, latest_date, data, benchmark = result

    # 获取新的Top12
    new_top12 = get_top12_signals(data, benchmark, latest_date)

    # 当前持仓
    current_positions = set(holdings['positions'].keys())
    new_positions = set(new_top12)

    # 计算买卖
    to_sell = current_positions - new_positions
    to_buy = new_positions - current_positions
    to_hold = current_positions & new_positions

    # 计算每只股票目标金额
    total_capital = holdings['total_capital']
    target_per_stock = total_capital / 12

    # 生成消息
    lines = []
    lines.append(f"📊 调仓指令 ({latest_date.strftime('%Y-%m-%d')})")
    lines.append(f"总资金: {total_capital:,.0f}元")
    lines.append("")

    # 卖出
    if to_sell:
        lines.append("【卖出】")
        for code in to_sell:
            pos = holdings['positions'].get(code, {})
            shares = pos.get('shares', 0)
            price = prices.get(code, 0)
            name = STOCK_NAMES.get(code, code)
            code_short = code.replace('sh.', '').replace('sz.', '')
            lines.append(f"• {name}({code_short}) 卖{shares}股 × {price:.2f}元")
        lines.append("")

    # 买入
    if to_buy:
        lines.append("【买入】")
        for code in to_buy:
            price = prices.get(code, 0)
            if price > 0:
                shares = int(target_per_stock / price / 100) * 100
                if shares < 100:
                    shares = 100
                name = STOCK_NAMES.get(code, code)
                code_short = code.replace('sh.', '').replace('sz.', '')
                lines.append(f"• {name}({code_short}) 买{shares}股 × {price:.2f}元")
        lines.append("")

    # 持仓不变
    if to_hold:
        lines.append(f"【持有不动】({len(to_hold)}只)")
        hold_names = [STOCK_NAMES.get(c, c) for c in to_hold]
        lines.append(", ".join(hold_names))

    if not to_sell and not to_buy:
        lines.append("无需调仓，继续持有")

    lines.append("")
    lines.append("请确认后回复「已执行」")

    return "\n".join(lines)


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='发送调仓信号到飞书')
    parser.add_argument('--test', action='store_true', help='测试发送')
    parser.add_argument('--signal', action='store_true', help='发送调仓信号')
    parser.add_argument('--text', type=str, help='发送自定义文本')
    parser.add_argument('--user', type=str, help='目标用户open_id')
    parser.add_argument('--chat', type=str, help='目标群聊chat_id')

    args = parser.parse_args()

    if args.test:
        msg = f"🤖 测试消息\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n量化交易系统运行正常"
        send_text(msg, user_id=args.user, chat_id=args.chat)

    elif args.signal:
        msg = generate_signal_message()
        print("\n" + "="*50)
        print(msg)
        print("="*50 + "\n")
        send_text(msg, user_id=args.user, chat_id=args.chat)

    elif args.text:
        send_text(args.text, user_id=args.user, chat_id=args.chat)

    else:
        # 默认生成并显示信号（不发送）
        msg = generate_signal_message()
        print(msg)
        print("\n使用 --signal 参数发送到飞书")


if __name__ == '__main__':
    main()
