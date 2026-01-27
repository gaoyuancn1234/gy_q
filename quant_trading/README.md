# 量化交易系统

一个功能丰富的量化交易系统，支持中国A股和港股的回测、模拟交易和真实交易。

## 功能特性

### 数据获取
- **A股数据**: 支持akshare（免费）、tushare、baostock等数据源
- **港股数据**: 支持akshare（免费）、富途API等数据源
- **数据类型**: 日K、周K、月K、分钟K线、逐笔成交等
- **本地存储**: SQLite数据库存储历史数据

### 交易策略
- **均线策略**: 双均线、三均线、MACD等
- **动量策略**: 动量因子、RSI、KDJ等
- **均值回归**: 布林带、Z-score等
- **因子策略**: 单因子、多因子、Alpha因子等
- **自定义策略**: 可继承基类实现自定义策略

### 回测引擎
- 完整的回测框架
- 考虑手续费、印花税、滑点
- 详细的绩效分析报告
- 支持多股票组合回测

### 风险管理
- 仓位控制（单股、行业、总仓位）
- 止损止盈管理
- 跟踪止损
- 最大回撤控制
- 单日亏损限制
- 冷却期管理

### 真实交易
- **A股**: 支持通过easytrader连接多家券商，支持QMT量化接口
- **港股**: 支持富途、长桥、盈透等券商
- **模拟交易**: 完整的模拟交易功能用于策略验证

### 监控告警
- 实时仪表板监控
- 多渠道告警（日志、邮件、钉钉、企业微信）
- 风险预警

## 目录结构

```
quant_trading/
├── config/             # 配置模块
│   ├── settings.py     # 全局设置
│   └── broker_config.py # 券商配置
├── data/               # 数据模块
│   ├── data_source.py  # 数据源基类
│   ├── a_stock_data.py # A股数据
│   ├── hk_stock_data.py # 港股数据
│   └── database.py     # 数据库操作
├── strategy/           # 策略模块
│   ├── base_strategy.py # 策略基类
│   ├── ma_strategy.py  # 均线策略
│   ├── momentum_strategy.py # 动量策略
│   ├── mean_reversion.py # 均值回归
│   └── factor_strategy.py # 因子策略
├── backtest/           # 回测模块
│   ├── engine.py       # 回测引擎
│   └── performance.py  # 绩效分析
├── risk/               # 风险管理
│   ├── position_manager.py # 仓位管理
│   └── risk_control.py # 风控规则
├── broker/             # 券商接口
│   ├── base_broker.py  # 券商基类
│   ├── a_stock_broker.py # A股券商
│   ├── hk_stock_broker.py # 港股券商
│   └── paper_trading.py # 模拟交易
├── monitor/            # 监控模块
│   ├── dashboard.py    # 仪表板
│   └── alerting.py     # 告警系统
├── utils/              # 工具模块
│   ├── logger.py       # 日志
│   └── helpers.py      # 辅助函数
├── examples/           # 示例代码
├── main.py             # 主程序
└── requirements.txt    # 依赖
```

## 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/quant_trading.git
cd quant_trading

# 安装依赖
pip install -r requirements.txt

# 可选：安装A股交易依赖
pip install easytrader

# 可选：安装港股交易依赖
pip install futu-api
pip install longbridge
```

## 快速开始

### 1. 回测示例

```python
from datetime import date
from quant_trading import (
    AStockDataSource,
    MAStrategy,
    BacktestEngine
)

# 连接数据源
data_source = AStockDataSource(source="akshare")
data_source.connect()

# 获取数据
df = data_source.get_kline("000001", "daily", date(2023,1,1), date(2023,12,31))
df['code'] = '000001'

# 创建策略
strategy = MAStrategy(short_period=5, long_period=20)

# 运行回测
engine = BacktestEngine(initial_capital=1000000)
result = engine.run(strategy, df)

# 打印结果
print(result.summary())
```

### 2. 模拟交易示例

```python
from quant_trading import (
    AStockDataSource,
    MAStrategy,
    PaperTradingBroker,
    RiskController
)

# 初始化组件
data_source = AStockDataSource()
data_source.connect()

broker = PaperTradingBroker(initial_cash=1000000)
broker.connect()

strategy = MAStrategy()
risk_controller = RiskController()

# 获取行情并生成信号
quotes = data_source.get_realtime_quote(['000001'])
# ... 生成信号并执行交易
```

### 3. 实盘交易示例

```python
# A股实盘（使用easytrader）
from quant_trading import EasyTraderBroker

broker = EasyTraderBroker(
    broker="universal_client",
    client_path=r"C:\同花顺\xiadan.exe"
)
broker.connect()

# 下单
order = broker.buy("000001", price=10.0, quantity=100)
```

```python
# 港股实盘（使用富途）
from quant_trading import FutuBroker

broker = FutuBroker(
    host="127.0.0.1",
    port=11111,
    trade_env="REAL"
)
broker.connect()

# 下单
order = broker.buy("00700", price=300.0, quantity=100)
```

### 4. 命令行使用

```bash
# 回测
python main.py --mode backtest --codes 000001 600000 --start 2023-01-01 --end 2023-12-31

# 模拟交易
python main.py --mode paper --codes 000001 600000 --strategy ma

# 实盘交易（谨慎使用）
python main.py --mode live --codes 000001 --strategy macd
```

## 策略开发

### 自定义策略

```python
from quant_trading import BaseStrategy, Signal, SignalType

class MyStrategy(BaseStrategy):
    def __init__(self, param1=10, param2=20):
        super().__init__("MyStrategy", {'param1': param1, 'param2': param2})

    def generate_signals(self, data):
        signals = []
        # 你的策略逻辑
        # ...
        if buy_condition:
            signals.append(Signal(
                code=data['code'].iloc[-1],
                signal_type=SignalType.BUY,
                price=data['close'].iloc[-1],
                strength=0.8,
                reason="买入原因"
            ))
        return signals
```

## 券商配置

### A股券商（easytrader）

```python
from quant_trading import EasyTraderBroker

# 同花顺
broker = EasyTraderBroker(
    broker="universal_client",
    client_path=r"C:\同花顺\xiadan.exe"
)

# 华泰
broker = EasyTraderBroker(
    broker="ht",
    client_path=r"C:\华泰\xiadan.exe"
)
```

### 港股券商

```python
# 富途
from quant_trading import FutuBroker
broker = FutuBroker(host="127.0.0.1", port=11111)

# 长桥
from quant_trading import LongbridgeBroker
broker = LongbridgeBroker(
    app_key="your_key",
    app_secret="your_secret",
    access_token="your_token"
)
```

## 风险提示

⚠️ **投资有风险，入市需谨慎**

- 本系统仅供学习和研究使用
- 实盘交易前请充分测试策略
- 请使用模拟交易进行充分验证
- 请合理控制仓位和风险
- 过往业绩不代表未来表现

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License
