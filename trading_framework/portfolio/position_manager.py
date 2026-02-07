#!/usr/bin/env python3
"""
持仓状态管理
- 记录当前持仓
- 计算调仓信号（买入/卖出）
- 止盈止损检查
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TOP_K, STOP_LOSS, TAKE_PROFIT


# ============================================
# 配置
# ============================================

# 持仓文件路径
POSITION_FILE = Path(__file__).parent.parent / 'data' / 'positions.json'

# 交易周期（天）- 超过此天数强制检查是否需要卖出
HOLDING_PERIOD = 5  # 与预测目标一致（未来5日收益）

# 卖出条件
SELL_CONDITIONS = {
    'not_in_top': True,      # 不在TopN中则卖出
    'pred_negative': True,   # 预测变负则卖出
    'stop_loss': True,       # 止损（默认-8%）
    'take_profit': True,     # 止盈（默认+30%）
    'max_holding_days': 20,  # 最大持有天数
}


class PositionManager:
    """持仓管理器"""

    def __init__(self, position_file: str = None):
        self.position_file = Path(position_file) if position_file else POSITION_FILE
        self.position_file.parent.mkdir(parents=True, exist_ok=True)
        self.positions: Dict[str, dict] = {}
        self._load_positions()

    def _load_positions(self):
        """加载持仓记录"""
        if self.position_file.exists():
            with open(self.position_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.positions = data.get('positions', {})
                print(f"加载持仓: {len(self.positions)}只股票")
        else:
            self.positions = {}

    def _save_positions(self):
        """保存持仓记录"""
        data = {
            'updated_at': datetime.now().isoformat(),
            'positions': self.positions
        }
        with open(self.position_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_position(self, code: str, buy_price: float, shares: int,
                     buy_date: str = None):
        """添加持仓"""
        self.positions[code] = {
            'code': code,
            'buy_price': buy_price,
            'shares': shares,
            'buy_date': buy_date or datetime.now().strftime('%Y-%m-%d'),
            'cost': buy_price * shares
        }
        self._save_positions()
        print(f"添加持仓: {code} {shares}股 @ {buy_price}")

    def remove_position(self, code: str, sell_price: float = None,
                        reason: str = None):
        """移除持仓"""
        if code in self.positions:
            pos = self.positions.pop(code)
            self._save_positions()

            if sell_price:
                profit = (sell_price - pos['buy_price']) / pos['buy_price']
                print(f"卖出: {code} 盈亏: {profit*100:.2f}% 原因: {reason}")
            return pos
        return None

    def update_position(self, code: str, current_price: float):
        """更新持仓状态（用于计算盈亏）"""
        if code in self.positions:
            pos = self.positions[code]
            pos['current_price'] = current_price
            pos['profit_pct'] = (current_price - pos['buy_price']) / pos['buy_price']
            pos['market_value'] = current_price * pos['shares']
            pos['updated_at'] = datetime.now().isoformat()
            self._save_positions()

    def get_positions(self) -> Dict[str, dict]:
        """获取所有持仓"""
        return self.positions.copy()

    def get_position(self, code: str) -> Optional[dict]:
        """获取单个持仓"""
        return self.positions.get(code)

    def holding_days(self, code: str) -> int:
        """计算持有天数"""
        if code not in self.positions:
            return 0
        buy_date = datetime.strptime(self.positions[code]['buy_date'], '%Y-%m-%d')
        return (datetime.now() - buy_date).days

    def generate_rebalance_signals(self, new_signals: List[dict],
                                   current_prices: Dict[str, float] = None
                                   ) -> Dict[str, List[dict]]:
        """
        生成调仓信号

        参数:
            new_signals: 新的预测信号 [{'code': 'sh.600519', 'pred': 0.05}, ...]
            current_prices: 当前价格 {'sh.600519': 100.0, ...}

        返回:
            {
                'buy': [{'code': ..., 'reason': ...}, ...],
                'sell': [{'code': ..., 'reason': ...}, ...],
                'hold': [{'code': ..., 'reason': ...}, ...]
            }
        """
        current_prices = current_prices or {}
        result = {'buy': [], 'sell': [], 'hold': []}

        # 新信号的股票代码和预测值
        new_codes = {s['code'] for s in new_signals}
        new_preds = {s['code']: s['pred'] for s in new_signals}

        # 当前持仓代码
        current_codes = set(self.positions.keys())

        # 1. 检查现有持仓 - 是否需要卖出
        for code, pos in self.positions.items():
            sell_reason = None
            current_price = current_prices.get(code, pos.get('current_price', pos['buy_price']))

            # 更新当前价格
            if current_price:
                self.update_position(code, current_price)
                profit_pct = (current_price - pos['buy_price']) / pos['buy_price']
            else:
                profit_pct = 0

            # 检查卖出条件
            if SELL_CONDITIONS['stop_loss'] and profit_pct <= -STOP_LOSS:
                sell_reason = f"止损 ({profit_pct*100:.1f}%)"

            elif SELL_CONDITIONS['take_profit'] and profit_pct >= TAKE_PROFIT:
                sell_reason = f"止盈 ({profit_pct*100:.1f}%)"

            elif SELL_CONDITIONS['not_in_top'] and code not in new_codes:
                sell_reason = "不在推荐列表中"

            elif SELL_CONDITIONS['pred_negative'] and code in new_preds and new_preds[code] < 0:
                sell_reason = f"预测转负 ({new_preds[code]:.4f})"

            elif SELL_CONDITIONS['max_holding_days']:
                days = self.holding_days(code)
                if days > SELL_CONDITIONS['max_holding_days']:
                    sell_reason = f"持有超过{days}天"

            if sell_reason:
                result['sell'].append({
                    'code': code,
                    'reason': sell_reason,
                    'buy_price': pos['buy_price'],
                    'current_price': current_price,
                    'profit_pct': profit_pct,
                    'holding_days': self.holding_days(code)
                })
            else:
                result['hold'].append({
                    'code': code,
                    'reason': '继续持有',
                    'pred': new_preds.get(code, 0),
                    'profit_pct': profit_pct,
                    'holding_days': self.holding_days(code)
                })

        # 2. 检查新信号 - 是否需要买入
        for sig in new_signals:
            code = sig['code']
            if code not in current_codes and sig['pred'] > 0:
                # 限制持仓数量
                current_hold = len(result['hold'])
                if current_hold < TOP_K:
                    result['buy'].append({
                        'code': code,
                        'reason': f"新入选 (预测: {sig['pred']:.4f})",
                        'pred': sig['pred']
                    })

        return result

    def get_summary(self) -> str:
        """获取持仓摘要"""
        if not self.positions:
            return "当前无持仓"

        lines = [f"📊 当前持仓 ({len(self.positions)}只)\n"]
        total_cost = 0
        total_value = 0

        for code, pos in self.positions.items():
            days = self.holding_days(code)
            profit_pct = pos.get('profit_pct', 0)
            emoji = "🔺" if profit_pct >= 0 else "🔻"

            lines.append(f"  {code}")
            lines.append(f"    成本: ¥{pos['buy_price']:.2f} × {pos['shares']}股")
            lines.append(f"    盈亏: {emoji} {profit_pct*100:.2f}%")
            lines.append(f"    持有: {days}天")
            lines.append("")

            total_cost += pos.get('cost', 0)
            total_value += pos.get('market_value', pos.get('cost', 0))

        if total_cost > 0:
            total_profit = (total_value - total_cost) / total_cost
            lines.append(f"总成本: ¥{total_cost:,.2f}")
            lines.append(f"总市值: ¥{total_value:,.2f}")
            lines.append(f"总盈亏: {'🔺' if total_profit >= 0 else '🔻'} {total_profit*100:.2f}%")

        return "\n".join(lines)


def test():
    """测试"""
    pm = PositionManager()

    # 模拟添加持仓
    pm.add_position('sh.600519', 1800.0, 100, '2026-01-20')
    pm.add_position('sz.000858', 150.0, 500, '2026-01-25')

    # 更新价格
    pm.update_position('sh.600519', 1850.0)
    pm.update_position('sz.000858', 145.0)

    print(pm.get_summary())

    # 模拟新信号
    new_signals = [
        {'code': 'sh.600519', 'pred': 0.03},
        {'code': 'sh.601012', 'pred': 0.05},
        {'code': 'sz.002415', 'pred': 0.02},
    ]

    result = pm.generate_rebalance_signals(new_signals)
    print("\n调仓信号:")
    print(f"  卖出: {result['sell']}")
    print(f"  买入: {result['buy']}")
    print(f"  持有: {result['hold']}")


if __name__ == '__main__':
    test()
