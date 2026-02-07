# 量化交易系统 - Claude Code 上下文

## 策略说明
- **策略**: 15日相对强度轮动
- **持仓**: Top 12 等权配置
- **调仓**: 每周五
- **验证**: 2023-2025三年均跑赢大盘

## 凭证管理
- 所有敏感信息存放于 `.env` 文件（已 .gitignore）
- 环境变量: `FEISHU_APP_ID_1/2`, `FEISHU_APP_SECRET_1/2`, `FEISHU_USER_OPEN_ID`

## 项目结构
```
trading_framework/
├── smart_bot.py           # 飞书机器人（多App、多用户隔离）
├── daemon.py              # 守护进程（自动重启）
├── signal_today.py        # 生成交易信号
├── send_signal.py         # 发送调仓指令到飞书
├── .env                   # 敏感凭证（不入 git）
├── CLAUDE.md              # 本文件
├── config/settings.py     # 策略参数、股票池配置
├── data/
│   ├── data_loader.py     # 行情数据加载
│   └── stock_pool.py      # 股票池定义
├── portfolio/
│   ├── trade_executor.py  # 调仓指令生成
│   ├── position_manager.py
│   └── holdings.json      # 当前持仓
├── memory/                # 用户记忆（按 open_id 隔离）
└── archive/               # 历史文件（旧版bot/优化实验等）
```

## 工作流程

### 1. 每周五收盘后生成调仓指令
```bash
python portfolio/trade_executor.py --action signal
```

### 2. 用户反馈成交情况后更新持仓
```bash
python portfolio/trade_executor.py --action update
```

## 启动/管理
```bash
python daemon.py          # 启动守护进程
python daemon.py stop     # 停止
python daemon.py status   # 查看状态
python daemon.py restart  # 重启
```
