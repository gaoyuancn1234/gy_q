"""QuantaAlpha 论文超参常量 (Section 4) — v3 对齐论文实现

对齐论文开源实现: https://github.com/QuantaAlpha/QuantaAlpha
"""

# --- 挖掘规模 (论文 experiment.yaml 默认值) ---
N_DIRECTIONS = 2        # 初始方向数 (论文 num_directions=2)
MAX_ROUNDS = 3          # 最大演化轮次 (论文 max_rounds=3)
N_CANDIDATES = 2        # 每个方向生成的候选因子数 (论文 n_candidates=2)
CROSSOVER_N = 2         # 每轮 crossover 组合数 (论文 crossover_n=2)
CROSSOVER_SIZE = 2      # 每次 crossover 的父代数 (论文 crossover_size=2)

# --- Trace 系统 (论文 core/proposal.py) ---
HISTORY_LIMIT = 6       # 历史轨迹 prompt 最大条数 (论文 DEFAULT_HISTORY_LIMIT=6)
MIN_HISTORY_LIMIT = 1   # 最少保留条数

# --- 因子再生 (论文 _convert_with_history_limit) ---
MAX_REGEN_ATTEMPTS = 5  # 因子被拒后的最大重试次数 (论文 max_regen=5)

# --- 组合回测 (论文 QlibFactorRunner.develop) ---
BACKTEST_ENABLED = True  # 是否开启每轮组合回测

# --- 表达式约束 (论文 Section 4) ---
MAX_SYMBOL_LENGTH = 200         # 表达式最大字符数 (论文 symbol_length_threshold=200)
MAX_BASE_FEATURES = 5           # 最大基础字段数 (论文 max_base_features=5)
MAX_FREE_PARAM_RATIO = 0.50     # 自由参数占比上限
MAX_NESTING_DEPTH = 5           # 最大嵌套深度

# --- 因子池准入 (论文 Section 4.3) ---
REDUNDANCY_CORR = 0.7           # 相关系数阈值
AST_SIMILARITY = 0.8            # AST 同构相似度阈值
POOL_CAP_RATIO = 0.50           # 池容量 = min(total * ratio, POOL_MAX)
POOL_MAX = 200                  # 池容量硬上限

# --- 回测参数 (论文 Table 5) ---
TOPK = 50
N_DROP = 5

# --- 时间区间 (论文 Table 5/7, 与 Exp 012 一致) ---
TRAIN_START = '2016-01-01'
TRAIN_END = '2020-12-31'
VALID_START = '2021-01-01'
VALID_END = '2021-12-31'
TEST_START = '2022-01-01'
TEST_END = '2025-12-26'

# --- 基础特征 ---
# 论文用 6 特征含 $vwap, 但我们的 Qlib 数据无 vwap.day.bin
# VWAP 需写为 Div($amount, $volume + 1e-8), 不能直接引用 $vwap
BASE_FEATURES = ['$open', '$high', '$low', '$close', '$volume', '$amount', '$turn']

# --- Claude CLI 配置 ---
CLAUDE_CLI = '/usr/local/bin/claude'
CLAUDE_TIMEOUT = 300  # 秒
MAX_RETRY = 10         # LLM 调用最大重试次数 (论文 max_retry=30, 实际用 10)
RETRY_WAIT = 5         # 重试等待秒数 (论文 retry_wait_seconds=15, 实际用 5)


# ============ Prompt 模板 (对齐论文 Jinja2 模板) ============

# --- LLM 反馈生成 (对齐论文 factor_feedback_generation) ---
FEEDBACK_PROMPT = """你是一位量化因子研究审核员。请对以下因子挖掘实验结果给出结构化反馈。

## 当前实验
- 假说: {hypothesis}
- 因子名: {factor_name}
- 因子表达式: {factor_expr}
- IC: {ic:.4f}, ICIR: {icir:.4f}, RankIC: {rank_ic:.4f}
{backtest_section}

## SOTA (当前最佳)
{sota_section}

## 任务
1. 观察当前实验结果的关键特征 (observations)
2. 评估假说的有效性 (hypothesis_evaluation)
3. 提出改进方向 (new_hypothesis)
4. 推理是否接受当前结果 (reasoning)
5. 最终决策: 当前结果是否比 SOTA 更好? (decision: true/false)

请输出 JSON:
```json
{{
  "observations": "对当前实验结果的关键观察 (2-3 句话)",
  "hypothesis_evaluation": "假说有效性评估 (1-2 句话)",
  "new_hypothesis": "改进方向建议 (1-2 句话)",
  "reasoning": "接受/拒绝的理由 (1-2 句话)",
  "decision": true/false
}}
```"""

# --- 单条历史记录渲染模板 (对齐论文 hypothesis_and_feedback) ---
TRACE_ENTRY_TMPL = """### 轮次 {round_idx}
- 假说: {hypothesis}
- 因子: {factor_name} = {factor_expr}
- IC={ic:.4f}, ICIR={icir:.4f}, RankIC={rank_ic:.4f}
{backtest_line}
- 反馈: {feedback_summary}
- 决策: {"接受 (新 SOTA)" if decision else "拒绝"}"""

# --- 因子被拒后的再生反馈模板 (对齐论文 expression_duplication) ---
REGEN_FEEDBACK_TMPL = """## 因子被拒原因
以下因子未通过复杂度/合法性检查，请重新生成:
{rejection_details}

## 已尝试的因子列表 (避免重复)
{tried_factors}

请生成全新的因子表达式，避免与上述因子重复。"""


def get_claude_env():
    """获取 Claude CLI 子进程的环境变量

    必须 unset CLAUDECODE, 否则嵌套调用会被拒绝:
    "Claude Code cannot be launched inside another Claude Code session"
    """
    import os
    env = os.environ.copy()
    env.pop('CLAUDECODE', None)
    return env

# --- Qlib 操作符参考 (供 prompt 使用) ---
QLIB_OPERATORS = """可用算子:
- 时序: Ref(x,N), Mean(x,N), Std(x,N), Sum(x,N), Delta(x,N), Min(x,N), Max(x,N), Slope(x,N), Rsquare(x,N)
- 截面: Rank(x,N) — 注意需要 window 参数 N
- 运算: Abs(x), Log(x), Sign(x), Power(x,N), Div(x,y), Greater(x,y), Less(x,y), If(cond,x,y)
- 统计: Corr(x,y,N), Cov(x,y,N) — 要求 x,y 日期范围一致 (同源字段)
- 注意: Max/Min 是时序滚动的; 截面比较用 Greater(x,y)/Less(x,y)"""

QLIB_CONSTRAINTS = """约束:
1. 除零保护: 分母加 1e-8，如 Div(x, y + 1e-8)
2. Corr/Cov 的两个字段必须同源 (都来自日频 OHLCV)
3. 不得使用未来数据 (Ref($close, -N) 中 N 必须 > 0)
4. 因子名全大写下划线格式
5. 表达式必须是有效的 Qlib 表达式
6. 表达式长度 ≤ 200 字符, 嵌套 ≤ 5 层, 基础字段 ≤ 5 种"""
