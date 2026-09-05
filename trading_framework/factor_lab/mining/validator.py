"""因子表达式验证 — 静态 + 动态

静态: 名称格式、括号配对、字段合法、无前视
动态: Qlib D.features() 小样本实际计算
"""
import re

# 合法字段
VALID_FIELDS = {
    "$open", "$high", "$low", "$close", "$volume", "$amount",
    "$turn", "$pe_ttm", "$pb", "$total_mv", "$circ_mv",
}

# 合法算子
VALID_OPS = {
    "Ref", "Mean", "Std", "Sum", "Delta", "Min", "Max", "Slope", "Rsquare",
    "Rank", "Abs", "Log", "Sign", "Power", "Div", "Greater", "Less", "If",
    "Corr", "Cov", "IdxMin", "IdxMax", "Quantile", "Mad", "Kurt", "Skew",
    "Mul", "Add", "Sub",
}

# 带时序位移/窗口参数的算子 —— 这些算子的数值参数为负即构成前视
_SHIFT_OPS = {
    "Ref", "Mean", "Std", "Sum", "Delta", "Min", "Max", "Slope", "Rsquare",
    "Rank", "Corr", "Cov", "IdxMin", "IdxMax", "Quantile", "Mad", "Kurt", "Skew",
}


def _split_args(inner: str) -> list[str]:
    """按顶层逗号切分参数，忽略嵌套括号内的逗号"""
    args, depth, cur = [], 0, []
    for ch in inner:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            args.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    args.append(''.join(cur))
    return [a.strip() for a in args]


def _find_negative_shift(expr: str):
    """找出任何时序算子的负数位移参数 (前视)

    用括号配对解析，不能用正则。
    2026-09-05: 原实现是 re.findall(r'Ref\\([^,]+,\\s*(-\\d+)\\)')，其中
    [^,]+ 不允许逗号 —— 只要 Ref 的第一个参数是含逗号的嵌套表达式就完全
    漏检。实测:
        Ref($close, -5)                      -> 拒绝 ✓
        Ref(Corr($close, $volume, 10), -5)   -> **通过** ✗
        Ref(Mean($close, 5), -3)             -> **通过** ✗
    而 LLM 生成的因子表达式绝大多数是嵌套的，等于这道前视闸对最常见的
    形式完全无效。用未来数据的因子 IC 会异常高、顺利通过 beat_baseline
    写进 mined.py —— 这可能正是 run_006/run_010 那 22 个因子样本内
    +7.4% 的来源之一。

    Returns: 命中的片段描述，无则 None
    """
    for m in re.finditer(r'\b([A-Z][a-zA-Z]*)\s*\(', expr):
        op = m.group(1)
        if op not in _SHIFT_OPS:
            continue
        # 从左括号开始做括号配对，取出完整参数串
        start = m.end()
        depth, i = 1, start
        while i < len(expr) and depth > 0:
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
            i += 1
        if depth != 0:
            continue                      # 括号不配对，前面已单独检查
        for arg in _split_args(expr[start:i - 1]):
            if re.fullmatch(r'-\s*\d+', arg):
                return f"{op}(..., {arg})"
    return None


def validate_expression(name: str, expr: str) -> tuple[bool, str]:
    """静态验证: 名称格式、括号配对、字段合法、无前视

    Returns:
        (is_valid, error_message)
    """
    # 名称格式
    if not re.match(r'^[A-Z][A-Z0-9_]+$', name):
        return False, f"名称 '{name}' 不符合全大写下划线格式"

    if len(name) < 3 or len(name) > 40:
        return False, f"名称长度 {len(name)} 不在 [3, 40] 范围"

    # 表达式非空
    if not expr or not expr.strip():
        return False, "表达式为空"

    # 括号配对
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            return False, "括号不匹配 (多余的右括号)"
    if depth != 0:
        return False, f"括号不匹配 (缺少 {depth} 个右括号)"

    # 字段合法性 — 提取所有 $xxx
    fields_used = set(re.findall(r'\$[a-z_]+', expr))
    invalid = fields_used - VALID_FIELDS
    if invalid:
        return False, f"不合法的字段: {invalid}"

    # 无前视 — 任何时序算子的位移/窗口参数不得为负
    bad = _find_negative_shift(expr)
    if bad:
        return False, f"检测到前视引用: {bad}"

    # 基本算子检查 — 提取所有 FuncName(
    ops_used = set(re.findall(r'([A-Z][a-zA-Z]+)\s*\(', expr))
    unknown = ops_used - VALID_OPS
    if unknown:
        return False, f"未知算子: {unknown}"

    # 复杂度约束
    ok, reason = check_complexity(expr)
    if not ok:
        return False, reason

    return True, ""


def check_complexity(expr: str) -> tuple[bool, str]:
    """检查表达式复杂度，防止过拟合

    规则:
    - 表达式长度 ≤ 200 字符
    - 括号嵌套深度 ≤ 5 层
    - 基础字段 ($xxx) ≤ 6 种

    Returns:
        (is_ok, reason) — 不通过时 reason 供 mutation 参考
    """
    # 长度
    if len(expr) > 200:
        return False, f"表达式过长 ({len(expr)} > 200 字符)，请简化"

    # 嵌套深度
    max_depth = 0
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    if max_depth > 5:
        return False, f"嵌套过深 ({max_depth} > 5 层)，请减少嵌套"

    # 基础字段数
    fields = set(re.findall(r'\$[a-z_]+', expr))
    if len(fields) > 6:
        return False, f"字段过多 ({len(fields)} > 6 种: {fields})，请精简"

    return True, ""


def validate_with_qlib(name: str, expr: str,
                       sample_days: int = 5) -> tuple[bool, str]:
    """动态验证: 用 D.features() 在小窗口实际计算

    捕获 Corr shape mismatch、除零、NaN 全空等问题。

    Returns:
        (is_valid, error_message)
    """
    try:
        from qlib.data import D
        import numpy as np

        inst = D.instruments("csi300")
        df = D.features(
            instruments=inst,
            fields=[expr],
            start_time="2025-12-01",
            end_time="2025-12-10",
        )

        if df is None or df.empty:
            return False, "D.features() 返回空"

        # 全 NaN 检查
        nan_ratio = df.iloc[:, 0].isna().mean()
        if nan_ratio > 0.95:
            return False, f"NaN 比例过高: {nan_ratio:.1%}"

        # inf 检查
        inf_count = np.isinf(df.iloc[:, 0].dropna()).sum()
        if inf_count > 0:
            return False, f"包含 {inf_count} 个 inf 值"

        return True, ""

    except Exception as e:
        err_msg = str(e)
        # 常见错误简化
        if "shape" in err_msg.lower():
            return False, "Corr/Cov shape mismatch (字段日期范围不一致)"
        return False, f"Qlib 计算失败: {err_msg[:200]}"
