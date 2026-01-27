"""
绩效分析模块
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import date
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化绩效分析器

        Args:
            risk_free_rate: 无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate

    def analyze(self, equity_curve: pd.DataFrame, trades: List = None) -> Dict[str, Any]:
        """
        分析绩效

        Args:
            equity_curve: 权益曲线DataFrame，需要包含 timestamp, total_value 列
            trades: 交易记录列表

        Returns:
            绩效指标字典
        """
        if equity_curve.empty:
            return {}

        df = equity_curve.copy()
        df['return'] = df['total_value'].pct_change()

        metrics = {}

        # 收益指标
        metrics['total_return'] = self.total_return(df)
        metrics['annual_return'] = self.annual_return(df)
        metrics['monthly_returns'] = self.monthly_returns(df)

        # 风险指标
        metrics['volatility'] = self.volatility(df)
        metrics['max_drawdown'] = self.max_drawdown(df)
        metrics['var_95'] = self.value_at_risk(df, 0.95)
        metrics['cvar_95'] = self.conditional_var(df, 0.95)

        # 风险调整收益
        metrics['sharpe_ratio'] = self.sharpe_ratio(df)
        metrics['sortino_ratio'] = self.sortino_ratio(df)
        metrics['calmar_ratio'] = self.calmar_ratio(df)
        metrics['information_ratio'] = self.information_ratio(df)

        # 交易统计
        if trades:
            metrics['trade_stats'] = self.trade_statistics(trades)

        return metrics

    def total_return(self, df: pd.DataFrame) -> float:
        """计算总收益率"""
        if df.empty:
            return 0
        return (df['total_value'].iloc[-1] - df['total_value'].iloc[0]) / df['total_value'].iloc[0]

    def annual_return(self, df: pd.DataFrame) -> float:
        """计算年化收益率"""
        total_ret = self.total_return(df)
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
        if days <= 0:
            return 0
        return (1 + total_ret) ** (365 / days) - 1

    def monthly_returns(self, df: pd.DataFrame) -> pd.Series:
        """计算月度收益率"""
        df = df.copy()
        df['month'] = df['timestamp'].dt.to_period('M')
        monthly = df.groupby('month')['total_value'].last().pct_change()
        return monthly

    def volatility(self, df: pd.DataFrame, annualize: bool = True) -> float:
        """计算波动率"""
        std = df['return'].std()
        if annualize:
            return std * np.sqrt(252)
        return std

    def max_drawdown(self, df: pd.DataFrame) -> float:
        """计算最大回撤"""
        df = df.copy()
        df['peak'] = df['total_value'].cummax()
        df['drawdown'] = (df['total_value'] - df['peak']) / df['peak']
        return abs(df['drawdown'].min())

    def value_at_risk(self, df: pd.DataFrame, confidence: float = 0.95) -> float:
        """计算VaR"""
        returns = df['return'].dropna()
        return np.percentile(returns, (1 - confidence) * 100)

    def conditional_var(self, df: pd.DataFrame, confidence: float = 0.95) -> float:
        """计算CVaR（条件VaR）"""
        returns = df['return'].dropna()
        var = self.value_at_risk(df, confidence)
        return returns[returns <= var].mean()

    def sharpe_ratio(self, df: pd.DataFrame) -> float:
        """计算夏普比率"""
        daily_rf = self.risk_free_rate / 252
        excess_returns = df['return'] - daily_rf
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def sortino_ratio(self, df: pd.DataFrame) -> float:
        """计算索提诺比率"""
        daily_rf = self.risk_free_rate / 252
        excess_returns = df['return'] - daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        if downside_returns.std() == 0:
            return 0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    def calmar_ratio(self, df: pd.DataFrame) -> float:
        """计算卡尔玛比率"""
        annual_ret = self.annual_return(df)
        max_dd = self.max_drawdown(df)
        if max_dd == 0:
            return 0
        return annual_ret / max_dd

    def information_ratio(self, df: pd.DataFrame, benchmark_return: float = 0.08) -> float:
        """计算信息比率"""
        daily_benchmark = benchmark_return / 252
        tracking_diff = df['return'] - daily_benchmark
        if tracking_diff.std() == 0:
            return 0
        return tracking_diff.mean() / tracking_diff.std() * np.sqrt(252)

    def trade_statistics(self, trades: List) -> Dict[str, Any]:
        """计算交易统计"""
        if not trades:
            return {}

        buy_trades = [t for t in trades if t.direction == "BUY"]
        sell_trades = [t for t in trades if t.direction == "SELL"]

        # 计算盈亏
        trade_pnl = []
        trade_returns = []

        for sell in sell_trades:
            matching_buys = [t for t in buy_trades
                           if t.code == sell.code and t.timestamp < sell.timestamp]
            if matching_buys:
                total_buy_qty = sum(t.quantity for t in matching_buys)
                total_buy_amount = sum(t.price * t.quantity for t in matching_buys)
                avg_buy_price = total_buy_amount / total_buy_qty

                pnl = (sell.price - avg_buy_price) * sell.quantity
                ret = (sell.price - avg_buy_price) / avg_buy_price

                trade_pnl.append(pnl)
                trade_returns.append(ret)

        winning_trades = [p for p in trade_pnl if p > 0]
        losing_trades = [p for p in trade_pnl if p <= 0]

        stats = {
            'total_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trade_pnl) if trade_pnl else 0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': min(losing_trades) if losing_trades else 0,
            'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else 0,
            'avg_trade_return': np.mean(trade_returns) if trade_returns else 0,
            'total_commission': sum(t.commission for t in trades),
            'total_stamp_duty': sum(t.stamp_duty for t in trades)
        }

        return stats

    def drawdown_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """回撤分析"""
        df = df.copy()
        df['peak'] = df['total_value'].cummax()
        df['drawdown'] = (df['total_value'] - df['peak']) / df['peak']

        # 找到所有回撤期
        in_drawdown = df['drawdown'] < 0
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)

        drawdowns = []
        start_idx = None

        for idx, row in df.iterrows():
            if drawdown_starts.loc[idx]:
                start_idx = idx
            elif drawdown_ends.loc[idx] and start_idx is not None:
                period = df.loc[start_idx:idx]
                drawdowns.append({
                    'start': period['timestamp'].iloc[0],
                    'end': period['timestamp'].iloc[-1],
                    'max_drawdown': abs(period['drawdown'].min()),
                    'duration': len(period)
                })
                start_idx = None

        return {
            'max_drawdown': abs(df['drawdown'].min()),
            'avg_drawdown': abs(df[df['drawdown'] < 0]['drawdown'].mean()) if any(df['drawdown'] < 0) else 0,
            'drawdown_periods': drawdowns,
            'longest_drawdown': max([d['duration'] for d in drawdowns]) if drawdowns else 0
        }

    def rolling_metrics(self, df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """计算滚动绩效指标"""
        df = df.copy()

        # 滚动收益率
        df['rolling_return'] = df['total_value'].pct_change(window)

        # 滚动波动率
        df['rolling_volatility'] = df['return'].rolling(window).std() * np.sqrt(252)

        # 滚动夏普比率
        daily_rf = self.risk_free_rate / 252
        excess = df['return'] - daily_rf
        df['rolling_sharpe'] = (
            excess.rolling(window).mean() /
            excess.rolling(window).std() *
            np.sqrt(252)
        )

        # 滚动最大回撤
        def rolling_max_dd(x):
            if len(x) == 0:
                return 0
            peak = x.cummax()
            dd = (x - peak) / peak
            return abs(dd.min())

        df['rolling_max_dd'] = df['total_value'].rolling(window).apply(rolling_max_dd, raw=False)

        return df

    def compare_strategies(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        比较多个策略的绩效

        Args:
            results: 策略结果列表，每个元素是一个包含绩效指标的字典

        Returns:
            比较结果DataFrame
        """
        comparison = pd.DataFrame(results)
        comparison = comparison.set_index('strategy_name')
        return comparison

    def generate_report(self, metrics: Dict[str, Any], output_format: str = "text") -> str:
        """
        生成绩效报告

        Args:
            metrics: 绩效指标字典
            output_format: 输出格式 (text, html, markdown)

        Returns:
            格式化的报告
        """
        if output_format == "text":
            return self._text_report(metrics)
        elif output_format == "markdown":
            return self._markdown_report(metrics)
        elif output_format == "html":
            return self._html_report(metrics)
        else:
            return self._text_report(metrics)

    def _text_report(self, metrics: Dict[str, Any]) -> str:
        """生成文本报告"""
        lines = [
            "=" * 50,
            "          绩效分析报告",
            "=" * 50,
            "",
            "【收益指标】",
            f"  总收益率: {metrics.get('total_return', 0):.2%}",
            f"  年化收益率: {metrics.get('annual_return', 0):.2%}",
            "",
            "【风险指标】",
            f"  年化波动率: {metrics.get('volatility', 0):.2%}",
            f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}",
            f"  VaR(95%): {metrics.get('var_95', 0):.2%}",
            f"  CVaR(95%): {metrics.get('cvar_95', 0):.2%}",
            "",
            "【风险调整收益】",
            f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}",
            f"  索提诺比率: {metrics.get('sortino_ratio', 0):.2f}",
            f"  卡尔玛比率: {metrics.get('calmar_ratio', 0):.2f}",
            "",
        ]

        if 'trade_stats' in metrics:
            stats = metrics['trade_stats']
            lines.extend([
                "【交易统计】",
                f"  总交易次数: {stats.get('total_trades', 0)}",
                f"  盈利次数: {stats.get('winning_trades', 0)}",
                f"  亏损次数: {stats.get('losing_trades', 0)}",
                f"  胜率: {stats.get('win_rate', 0):.2%}",
                f"  盈亏比: {stats.get('profit_factor', 0):.2f}",
                f"  平均盈利: {stats.get('avg_win', 0):.2f}",
                f"  平均亏损: {stats.get('avg_loss', 0):.2f}",
                f"  最大单笔盈利: {stats.get('largest_win', 0):.2f}",
                f"  最大单笔亏损: {stats.get('largest_loss', 0):.2f}",
            ])

        lines.append("=" * 50)
        return "\n".join(lines)

    def _markdown_report(self, metrics: Dict[str, Any]) -> str:
        """生成Markdown报告"""
        lines = [
            "# 绩效分析报告",
            "",
            "## 收益指标",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 总收益率 | {metrics.get('total_return', 0):.2%} |",
            f"| 年化收益率 | {metrics.get('annual_return', 0):.2%} |",
            "",
            "## 风险指标",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 年化波动率 | {metrics.get('volatility', 0):.2%} |",
            f"| 最大回撤 | {metrics.get('max_drawdown', 0):.2%} |",
            f"| VaR(95%) | {metrics.get('var_95', 0):.2%} |",
            "",
            "## 风险调整收益",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 夏普比率 | {metrics.get('sharpe_ratio', 0):.2f} |",
            f"| 索提诺比率 | {metrics.get('sortino_ratio', 0):.2f} |",
            f"| 卡尔玛比率 | {metrics.get('calmar_ratio', 0):.2f} |",
        ]

        return "\n".join(lines)

    def _html_report(self, metrics: Dict[str, Any]) -> str:
        """生成HTML报告"""
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; max-width: 600px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>绩效分析报告</h1>

            <h2>收益指标</h2>
            <table>
                <tr><th>指标</th><th>数值</th></tr>
                <tr><td>总收益率</td><td>{metrics.get('total_return', 0):.2%}</td></tr>
                <tr><td>年化收益率</td><td>{metrics.get('annual_return', 0):.2%}</td></tr>
            </table>

            <h2>风险指标</h2>
            <table>
                <tr><th>指标</th><th>数值</th></tr>
                <tr><td>年化波动率</td><td>{metrics.get('volatility', 0):.2%}</td></tr>
                <tr><td>最大回撤</td><td>{metrics.get('max_drawdown', 0):.2%}</td></tr>
                <tr><td>VaR(95%)</td><td>{metrics.get('var_95', 0):.2%}</td></tr>
            </table>

            <h2>风险调整收益</h2>
            <table>
                <tr><th>指标</th><th>数值</th></tr>
                <tr><td>夏普比率</td><td>{metrics.get('sharpe_ratio', 0):.2f}</td></tr>
                <tr><td>索提诺比率</td><td>{metrics.get('sortino_ratio', 0):.2f}</td></tr>
                <tr><td>卡尔玛比率</td><td>{metrics.get('calmar_ratio', 0):.2f}</td></tr>
            </table>
        </body>
        </html>
        """
