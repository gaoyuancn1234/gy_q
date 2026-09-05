"""QuantaAlpha 论文超参常量 (Section 4) — v4 论文规模实验

对齐论文开源实现: https://github.com/QuantaAlpha/QuantaAlpha
"""
from cli_paths import CLAUDE_BIN, GEMINI_BIN, CODEX_BIN

# --- 挖掘规模 (v4: 放大搜索空间) ---
N_DIRECTIONS = 10       # 初始方向数 (v4: 10, 覆盖更广搜索空间)
MAX_ROUNDS = 5          # 最大演化轮次 (v4: 5, 4轮演化 M→C→M→C)
N_CANDIDATES = 2        # 每个方向生成的候选因子数 (论文 n_candidates=2)
CROSSOVER_N = 3         # 每轮 crossover 组合数 (v4: 3, 利用10方向多样性)
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

# --- 语义一致性闸门 (论文核心组件之一) ---
#
# 论文摘要: "enforces semantic consistency across hypothesis, factor expression,
# and executable code"。原实现 verify_consistency() 写好了却从未被调用，
# consistency_ok 字段也从未赋值 —— 整道闸是死代码 (2026-09-02 修复)。
#
# 缺这道闸的后果与已记录的事故吻合: run_006/run_010 的 22 个因子样本内 +7.4%、
# 样本外 -12.9%。假说与表达式对不上的因子只能靠样本内噪声取胜。
CONSISTENCY_CHECK_ENABLED = True
CONSISTENCY_MIN_SCORE = 0.7     # score >= 此值视为一致 (原硬编码 0.7)
CONSISTENCY_MAX_REPAIR = 2      # 判定不一致后的定向重构次数 (论文 check_and_correct)
# 验证本身失败(调用异常/输出无法解析)时是否拦截。
# False = 放行但显式告警; True = 一律拦截。默认 False: 避免 LLM 偶发故障
# 卡死整条挖掘流水线，但绝不把"没验成"伪装成"验过了"。
CONSISTENCY_STRICT = False

# --- 因子池准入 (论文 Section 4.3) ---
REDUNDANCY_CORR = 0.7           # 相关系数阈值
AST_SIMILARITY = 0.8            # AST 同构相似度阈值

# 数值相关性准入闸门 (2026-09-03 接线)
# 此前 FactorPool.add() 只做 AST 相似度，check_corr_redundancy() 写好却从未被调用，
# 论文用来抑制因子拥挤的核心规则形同虚设。
CORR_CHECK_ENABLED = True
# 检查跑不起来时是否拒绝准入。False = 放行但显式告警; True = 一律拒绝。
# 默认 False 以免 qlib 未初始化时卡死整条挖掘流水线，但绝不静默当成"检查通过"。
CORR_CHECK_STRICT = False
POOL_CAP_RATIO = 0.50           # 池容量 = min(total * ratio, POOL_MAX)
POOL_MAX = 200                  # 池容量硬上限

# --- 因子质量筛选 (写入 mined.py 的门槛, 高于入池门槛) ---
QUALITY_MIN_ABS_ICIR = 0.4      # 略低于 evaluator "promising" 门槛 (0.5), LightGBM 可学方向
QUALITY_MIN_ABS_RANK_IC = 0.02  # 过滤纯噪声因子

# --- 回测参数 ---
# ⚠ 这两个常量【实际未生效】(2026-09-03 核实)。
# eval_agent 走的是 run_rolling_benchmark.run_backtest，用的是那边从
# signal_config.yaml 读来的生产参数 (当前 topk=8 / n_drop=2)。
#
# 保留论文值仅作参考。挖掘的评估口径必须与生产一致 —— 用 TopK 50 评出来的
# 因子，在生产的 TopK 8 上未必有增量。若哪天真要按论文口径评估，
# 需显式传参给 run_backtest，而不是改这里(改了也不生效)。
PAPER_TOPK = 50      # 论文 Table 5 值，仅存档
PAPER_N_DROP = 5
TOPK = PAPER_TOPK    # 向后兼容: eval_agent 仍 import 这两个名字
N_DROP = PAPER_N_DROP

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

# --- Rolling Eval (对齐生产 SOTA: D_expand_3v_3r 全窗口) ---
#
# ⚠ 挖掘评估期必须与最终验收期分离 (2026-08-30 修复)
#
# 旧配置 ROLLING_EVAL_TEST_END = '2026-02-05' 与 run_rolling_benchmark.TEST_END
# 完全相同，意味着挖掘在"最终评测的同一段数据"上筛选因子 —— 被选中的因子
# 必然在该段样本内占优，"跑赢基线"的判定失去意义。
#
# 实证后果 (run_006/run_010 挖出的 22 个因子):
#   样本内 (= 挖掘评估期)  Sharpe 1.643 → 1.765  (+7.4%)
#   样本外 (挖掘未见过)    Sharpe 0.351 → 0.306  (-12.9%)
# 符号反转 → 已全部从 mined.py 移除。
#
# 现在: 挖掘只在 MINING 区间内评估；MINING_HOLDOUT_START 之后的数据留作
# 验收，挖掘过程绝不可触碰。接受因子前须在 holdout 上复核:
#   python -m factor_lab.run_rolling_benchmark \
#       --presets alpha158_val alpha158_val_mined \
#       --test-start 2026-02-06 --test-end 2026-08-21 --tag holdout
ROLLING_EVAL_LITE = True          # 启用 rolling eval (替代单次训练回测)
ROLLING_EVAL_WINDOWS = 20         # 窗口数上限 (设足够大, 实际由 test period 决定)
ROLLING_EVAL_CONFIG = "D_expand_3v_3r"  # 对齐生产 SOTA (扩展窗口)
ROLLING_EVAL_TEST_START = '2024-01-01'
ROLLING_EVAL_TEST_END = '2026-02-05'   # 挖掘评估期上界 (不含之后数据)

# 验收 holdout: 挖掘不得使用此日期及之后的任何数据
MINING_HOLDOUT_START = '2026-02-06'

# 导入时自检: 挖掘评估期若越过 holdout 边界, 立刻失败而不是静默产生
# 样本内结论 (这正是 2026-08-30 之前的状态)
if ROLLING_EVAL_TEST_END >= MINING_HOLDOUT_START:
    raise ValueError(
        f"挖掘评估期 ROLLING_EVAL_TEST_END={ROLLING_EVAL_TEST_END} 越过了验收 "
        f"holdout 边界 MINING_HOLDOUT_START={MINING_HOLDOUT_START}。\n"
        "这会让挖掘在验收数据上筛选因子，'跑赢基线'的判定将失去意义。"
    )

# --- Evolved Mining (演化式挖掘) ---
EVOLVED_N_DIRECTIONS = 8          # 搜索方向数 (smoke: 3)
EVOLVED_N_ROUNDS = 6              # 演化轮次 (smoke: 2)  0=orig, 1=mut, 2=cross, 3=mut, 4=cross, 5=mut
ACCUMULATED_EVAL = True           # Step 4 Backtest 时包含 FactorPool 全局因子 (跨方向累积)

# --- Daily Session (每日渐进式挖掘) ---
DAILY_TOTAL_TIMEOUT = 5 * 3600   # 每日 session 总超时 (5h, 22:00→03:00)
DAILY_N_DIRECTIONS = 5            # 每次规划方向数
DAILY_BREADTH_STEPS = 5           # 广度阶段每个方向的步数
DAILY_EXHAUSTED_FAILURES = 3     # 连续失败 N 次标记方向为 exhausted

# --- FactorMiner 多阶段评估参数 (论文 Section 3.3-3.4) ---
USE_MULTISTAGE = True           # 启用多阶段评估 (--daily only)
FAST_IC_THRESHOLD = 0.02        # Stage 1 快速筛选阈值 (日频, 论文10min用0.04)
FAST_IC_PERIOD = ("2025-01-01", "2025-12-31")  # 快速筛选时间窗口
CORRELATION_THRESHOLD = 0.5     # Stage 2 相关性阈值 (论文值)
REPLACE_IC_MIN = 0.03           # Stage 2.5 替换最低 IC (日频, 论文10min用0.10)
REPLACE_RATIO = 1.3             # Stage 2.5 替换倍数 (论文值)
BATCH_DEDUP_AST = 0.7           # Stage 3 批内去重 AST 阈值

# --- Phase D: Importance 筛选 ---
IMPORTANCE_SCREEN_ENABLED = True    # 启用 importance 筛选
IMPORTANCE_MIN_THRESHOLD = 0       # importance > 0 即保留 (模型至少用过一次)

# --- Claude CLI 配置 ---
CLAUDE_CLI = CLAUDE_BIN
CLAUDE_TIMEOUT = 600  # 秒 (一致性验证等复杂 prompt 可能需要较长时间)
MAX_RETRY = 10         # LLM 调用最大重试次数 (论文 max_retry=30, 实际用 10)
RETRY_WAIT = 5         # 重试等待秒数 (论文 retry_wait_seconds=15, 实际用 5)

# --- 多 LLM 配置 (仅假设生成) ---
LLM_PROVIDERS = [
    {
        "name": "claude",
        "cli_path": CLAUDE_BIN,
        "cli_args": ["--print", "--dangerously-skip-permissions",
                     "--output-format", "text"],
        "prompt_flag": "-p",
        "env_remove": ["CLAUDECODE"],
        "timeout": 600,
        "daily_quota": 100,
    },
    # gemini / codex 默认禁用 —— 2026-09-05:
    # gemini CLI 已安装(v0.58.0)且 cli_paths 能探测到，但**未完成认证**，
    # 调用返回 exit 41 "Please set an Auth method"。不禁用的话轮询会真的
    # 派发给它、失败重试后再降级回 claude，白耗时间和日志。
    #
    # 启用步骤:
    #   1. 在交互式终端跑一次 `gemini`，走 Google OAuth 登录(用 Gemini Pro
    #      订阅账号，不额外计费；API Key 是另一套按量计费额度)
    #   2. 实测 `gemini -p "<真实长度的假说 prompt>"` 能返回，确认长 prompt
    #      非交互调用不会挂起 —— 参考 net_guard 那批教训: 第三方调用挂起
    #      而不抛异常时，降级分支永不执行
    #   3. 把下面的 "enabled" 改为 True
    {
        "name": "gemini",
        "cli_path": GEMINI_BIN,
        "cli_args": [],
        "prompt_flag": "-p",
        "env_remove": [],
        "timeout": 600,
        "daily_quota": 50,
        "enabled": False,
    },
    {
        "name": "codex",
        "cli_path": CODEX_BIN,
        "cli_args": ["exec", "--ephemeral"],
        "prompt_flag": "",
        "output_mode": "file",   # codex exec -o tmpfile 输出最终回复
        "env_remove": [],
        "timeout": 600,
        "daily_quota": 50,
        "enabled": False,        # codex CLI 未安装 (cli_paths 回退为裸名字)
    },
]

LLM_HYPOTHESIS_ONLY = True  # 多 LLM 仅用于假设生成

# --- 漏斗策略 (Phase D 多池并行评估) ---
FUNNEL_STRATEGIES = {
    "top_icir_10":   {"method": "top_icir",    "n": 10},
    "top_icir_20":   {"method": "top_icir",    "n": 20},
    "importance_10": {"method": "importance",  "n": 10},
    "importance_20": {"method": "importance",  "n": 20},
    "diverse_05":    {"method": "diverse",     "corr": 0.5, "n": 15},
    "consensus":     {"method": "consensus",   "min_votes": 3},
}
FUNNEL_N_BACKTEST = 2  # 漏斗: 评分最高的 N 个池做全量回测

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
- 决策: {decision_text}"""

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
- 时序: Ref(x,N), Mean(x,N), Std(x,N), Sum(x,N), Delta(x,N), Min(x,N), Max(x,N), Slope(x,N), Rsquare(x,N), IdxMax(x,N), IdxMin(x,N)
- 截面: Rank(x,N) — 注意需要 window 参数 N
- 运算: Abs(x), Log(x), Sign(x), Power(x,N), Div(x,y), Mul(x,y), Add(x,y), Sub(x,y), Greater(x,y), Less(x,y), Ge(x,y), Le(x,y), If(cond,x,y)
- 统计: Corr(x,y,N), Cov(x,y,N) — 要求 x,y 日期范围一致 (同源字段)
- 注意: Max/Min 是时序滚动的; 截面比较用 Greater(x,y)/Less(x,y)
- 别名: Neg(x)→Mul(x,-1), SMA→Mean, TsMax→Max, TsMin→Min, IfElse→If, GreaterEqual→Ge"""

# --- 算子别名映射 (供 prompt 引导 LLM 使用正确语法) ---
OPERATOR_ALIASES = {
    "Neg(x)": "Mul(x, -1)",
    "Inv(x)": "Div(1, x + 1e-8)",
    "Sqrt(x)": "Power(x, 0.5)",
    "Square(x)": "Power(x, 2)",
    "SMA(x, N)": "Mean(x, N)",
    "TsMax(x, N)": "Max(x, N)",
    "TsMin(x, N)": "Min(x, N)",
    "TsArgMax(x, N)": "IdxMax(x, N)",
    "TsArgMin(x, N)": "IdxMin(x, N)",
    "IfElse(c, x, y)": "If(c, x, y)",
    "GreaterEqual(x, y)": "Ge(x, y)",
    "LessEqual(x, y)": "Le(x, y)",
}

QLIB_CONSTRAINTS = """约束:
1. 除零保护: 分母加 1e-8，如 Div(x, y + 1e-8)
2. Corr/Cov 的两个字段必须同源 (都来自日频 OHLCV)
3. 不得使用未来数据 (Ref($close, -N) 中 N 必须 > 0)
4. 因子名全大写下划线格式
5. 表达式必须是有效的 Qlib 表达式
6. 表达式长度 ≤ 200 字符, 嵌套 ≤ 5 层, 基础字段 ≤ 5 种"""
