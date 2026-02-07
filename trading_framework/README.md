# 量化交易飞书机器人

基于飞书 WebSocket 长连接的智能量化交易助手，集成 Claude CLI 实现自然语言交互。

## 功能

- **自然语言交互** - 直接对话完成任务，无需记命令
- **调仓信号生成** - 15日相对强度轮动策略，Top 12 等权配置
- **图片分析** - 发送持仓截图，自动识别并更新持仓
- **画图/数据分析** - "画一下黄金价格曲线"、"分析一下沪深300走势"
- **多App支持** - 同时运行多个飞书应用，按 open_id 隔离用户会话
- **记忆系统** - 向量检索（Faiss）长期记忆 + 短期对话上下文
- **Redis临时文件** - 用户发送的图片等临时文件自动存入 Redis（90天），磁盘零残留
- **消息队列** - 批量消息智能合并，支持撤回已排队的任务
- **守护进程** - 自动监控和重启

## 快速开始

### 1. 创建飞书应用

1. 打开 [飞书开放平台](https://open.feishu.cn/app)，点击「创建企业自建应用」
2. 填写应用名称和描述，创建完成后进入应用详情页
3. 记录 **App ID** 和 **App Secret**（凭证与基础信息页面）

### 2. 开启机器人能力

1. 进入应用详情 → 「添加应用能力」 → 选择「机器人」
2. 进入「事件与回调」→「事件配置」，添加以下事件：
   - `im.message.receive_v1` — 接收消息
   - `im.message.message_read_v1` — 消息已读（可选）
3. 在「事件配置」中，Subscription Mode 选择 **使用长连接接收事件**（非 webhook）

### 3. 配置权限

进入「权限管理」，搜索并开通以下权限：

| 权限 | 权限标识 | 用途 |
|------|---------|------|
| 获取与发送单聊、群组消息 | `im:message` | 收发消息 |
| 获取用户发给机器人的单聊消息 | `im:message.receive_v1` | 接收消息事件 |
| 以应用的身份发消息 | `im:message:send_as_bot` | 主动发送消息 |
| 获取消息中的图片资源 | `im:resource` | 下载用户发送的图片 |
| 上传图片 | `im:image` | 发送图片消息 |
| 获取群组信息 | `im:chat:readonly` | 群聊场景（可选） |
| 添加消息表情回复 | `im:message.reactions:write` | 消息表情反馈 |

### 4. 发布应用

1. 进入「版本管理与发布」，创建版本并提交审核
2. 管理员在飞书管理后台审批通过
3. 发布后，在飞书中搜索机器人名称即可开始对话

### 5. 安装依赖

```bash
# Python 依赖
pip install lark-oapi python-dotenv redis numpy faiss-cpu onnxruntime tokenizers huggingface_hub

# Redis（macOS）
brew install redis
brew services start redis

# Claude CLI（需要单独安装）
# https://docs.anthropic.com/en/docs/claude-code
```

### 6. 配置凭证

复制 `.env.example` 或直接创建 `.env`：

```bash
# 飞书应用凭证（必填，支持多个App）
FEISHU_APP_ID_1=cli_xxxxxxxx
FEISHU_APP_SECRET_1=xxxxxxxx

# 第二个App（可选）
FEISHU_APP_ID_2=cli_yyyyyyyy
FEISHU_APP_SECRET_2=yyyyyyyy

# 已知用户（用于旧记忆迁移，可选）
FEISHU_USER_OPEN_ID=ou_xxxxxxxx

# Redis（可选，默认 localhost:6379）
# REDIS_URL=redis://localhost:6379/0
TEMP_FILE_TTL=7776000  # 临时文件保留时长（秒），默认90天
```

### 7. 启动

```bash
cd trading_framework

# 守护进程模式（推荐，自动重启）
python daemon.py

# 管理命令
python daemon.py status   # 查看状态
python daemon.py stop     # 停止
python daemon.py restart  # 重启

# 直接运行（调试用）
python smart_bot.py
```

## 使用方式

在飞书中与机器人私聊：

### 快捷命令

| 发送 | 功能 |
|------|------|
| `帮助` | 查看命令列表 |
| `信号` / `调仓` | 生成本周调仓指令（需确认） |
| `持仓` | 查看当前持仓 |
| `资金10万` | 设置总资金 |
| `历史文件` | 查看 Redis 中的历史图片/文件 |
| `取消` / `停止` | 终止当前正在执行的任务 |

### 自然语言

直接用自然语言对话，机器人会通过 Claude 理解并执行：

- "画一下黄金近一年的价格曲线"
- "分析一下我的持仓，有什么建议？"
- "帮我写一个计算夏普比率的脚本"
- 发送持仓截图 → 自动识别并更新持仓数据

### 操作确认

涉及执行命令的操作会弹出确认卡片，点击「允许」后才会执行，防止误操作。

## 项目结构

```
trading_framework/
├── smart_bot.py           # 飞书机器人主程序
├── daemon.py              # 守护进程
├── redis_files.py         # Redis 文件查询工具
├── signal_today.py        # 生成交易信号
├── send_signal.py         # 发送信号到飞书
├── .env                   # 凭证配置（不入 git）
├── config/
│   └── settings.py        # 策略参数、股票池
├── data/
│   ├── data_loader.py     # 行情数据加载
│   └── stock_pool.py      # 股票池定义
├── portfolio/
│   ├── trade_executor.py  # 调仓指令生成
│   ├── position_manager.py
│   └── holdings.json      # 当前持仓
├── memory/                # 用户记忆数据（按 open_id 隔离）
└── archive/               # 历史归档文件
```

## 架构说明

### 多App运行

通过 `.env` 中的 `FEISHU_APP_ID_1/2/3...` 配置多个飞书应用。启动时为每个 App 创建独立的 WebSocket 连接，共享同一个事件循环（使用 SDK 内部的 module-level event loop）。

### 用户隔离

每个用户（按飞书 open_id 区分）拥有独立的：
- 记忆系统（短期对话 + 长期向量记忆）
- 消息队列
- Claude 进程
- pending actions

### Redis 临时文件

用户发送的图片等文件处理完成后自动存入 Redis（默认保留90天），磁盘文件随即删除。需要时可通过文件 ID 从 Redis 取回。机器人的 Claude 实例知道如何查询和提取这些文件。

### 消息队列与撤回

当机器人正在处理任务时，新消息进入 pending 队列。任务完成后，若 pending 中有多条消息，会调用 Claude（JSON Schema 强制输出）判断哪些消息已被后续消息撤回，只执行有效的任务。
