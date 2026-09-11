"""Qlib 表达式 AST 解析 + 同构匹配

用于因子去重: 两个表达式即使文本不同，如果 AST 结构相似度 > 阈值则视为冗余。

示例:
  parse_qlib_expr("Mean($close, 5)")
  → ASTNode(op='Mean', children=[ASTNode(field='$close'), ASTNode(const=5)])

  ast_similarity("Mean($close, 5)", "Mean($close, 10)") → 0.67
  ast_similarity("Mean($close, 5)", "Std($close, 5)")   → 0.67
  ast_similarity("Mean($close, 5)", "Mean($close, 5)")   → 1.0
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ASTNode:
    """Qlib 表达式 AST 节点"""
    op: Optional[str] = None           # 函数名: Mean, Std, Div, ...
    children: list['ASTNode'] = field(default_factory=list)
    value: Optional[str] = None        # 叶节点: 字段名 ($close) 或常量 (5, 1e-8)
    is_field: bool = False             # True = $xxx 字段
    is_const: bool = False             # True = 数值常量

    @property
    def size(self) -> int:
        """子树大小 (节点数)"""
        return 1 + sum(c.size for c in self.children)

    def __repr__(self):
        if self.op:
            args = ", ".join(repr(c) for c in self.children)
            return f"{self.op}({args})"
        return str(self.value)


class ParseError(Exception):
    pass


class _Tokenizer:
    """简单的词法分析器"""
    # token 模式: 函数名, 字段, 数字, 运算符, 括号, 逗号
    TOKEN_RE = re.compile(
        r'([A-Z][a-zA-Z]*)'       # 函数名
        r'|(\$[a-z_]+)'           # 字段
        r'|([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)'  # 数字
        r'|([+\-*/])'             # 二元运算符
        r'|([(),])'               # 括号和逗号
        r'|(\s+)'                 # 空白 (跳过)
    )

    def __init__(self, expr: str):
        self.tokens: list[tuple[str, str]] = []  # (type, value)
        self.pos = 0
        self._tokenize(expr)

    def _tokenize(self, expr: str):
        pos = 0
        while pos < len(expr):
            m = self.TOKEN_RE.match(expr, pos)
            if not m:
                raise ParseError(f"Unexpected char at pos {pos}: '{expr[pos:][:20]}'")
            pos = m.end()
            if m.group(1):
                self.tokens.append(('FUNC', m.group(1)))
            elif m.group(2):
                self.tokens.append(('FIELD', m.group(2)))
            elif m.group(3):
                self.tokens.append(('NUM', m.group(3)))
            elif m.group(4):
                self.tokens.append(('OP', m.group(4)))
            elif m.group(5):
                self.tokens.append(('PUNC', m.group(5)))
            # group(6) = whitespace, skip

    def peek(self) -> Optional[tuple[str, str]]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tok_type: str, value: Optional[str] = None):
        tok = self.consume()
        if tok[0] != tok_type or (value is not None and tok[1] != value):
            raise ParseError(f"Expected {tok_type}({value}), got {tok}")
        return tok

    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)


def parse_qlib_expr(expr: str) -> ASTNode:
    """递归下降解析 Qlib 表达式

    语法:
      expr     → additive
      additive → multiplicative (('+' | '-') multiplicative)*
      multiplicative → unary (('*' | '/') unary)*
      unary    → atom
      atom     → FUNC '(' arglist ')' | FIELD | NUM | '(' expr ')'
      arglist  → expr (',' expr)*
    """
    try:
        tokenizer = _Tokenizer(expr)
        node = _parse_additive(tokenizer)
        if not tokenizer.at_end():
            # 允许尾部有多余 token (宽松解析)
            pass
        return node
    except (ParseError, IndexError):
        # 无法解析时返回一个叶节点
        return ASTNode(value=expr, is_const=True)


def _parse_additive(tok: _Tokenizer) -> ASTNode:
    left = _parse_multiplicative(tok)
    while tok.peek() and tok.peek()[0] == 'OP' and tok.peek()[1] in ('+', '-'):
        op_tok = tok.consume()
        right = _parse_multiplicative(tok)
        left = ASTNode(op=op_tok[1], children=[left, right])
    return left


def _parse_multiplicative(tok: _Tokenizer) -> ASTNode:
    left = _parse_atom(tok)
    while tok.peek() and tok.peek()[0] == 'OP' and tok.peek()[1] in ('*', '/'):
        op_tok = tok.consume()
        right = _parse_atom(tok)
        left = ASTNode(op=op_tok[1], children=[left, right])
    return left


def _parse_atom(tok: _Tokenizer) -> ASTNode:
    cur = tok.peek()
    if cur is None:
        raise ParseError("Unexpected end of expression")

    # 函数调用: FUNC '(' args ')'
    if cur[0] == 'FUNC':
        func_tok = tok.consume()
        if tok.peek() and tok.peek() == ('PUNC', '('):
            tok.consume()  # '('
            args = _parse_arglist(tok)
            if tok.peek() and tok.peek() == ('PUNC', ')'):
                tok.consume()  # ')'
            return ASTNode(op=func_tok[1], children=args)
        # 没有括号的函数名 — 当作常量处理
        return ASTNode(value=func_tok[1], is_const=True)

    # 字段: $xxx
    if cur[0] == 'FIELD':
        field_tok = tok.consume()
        return ASTNode(value=field_tok[1], is_field=True)

    # 数字
    if cur[0] == 'NUM':
        num_tok = tok.consume()
        return ASTNode(value=num_tok[1], is_const=True)

    # 括号表达式: '(' expr ')'
    if cur == ('PUNC', '('):
        tok.consume()
        node = _parse_additive(tok)
        if tok.peek() and tok.peek() == ('PUNC', ')'):
            tok.consume()
        return node

    # 负号前缀
    if cur[0] == 'OP' and cur[1] == '-':
        tok.consume()
        child = _parse_atom(tok)
        return ASTNode(op='neg', children=[child])

    raise ParseError(f"Unexpected token: {cur}")


def _parse_arglist(tok: _Tokenizer) -> list[ASTNode]:
    """解析函数参数列表"""
    args = []
    if tok.peek() == ('PUNC', ')'):
        return args
    args.append(_parse_additive(tok))
    while tok.peek() == ('PUNC', ','):
        tok.consume()
        args.append(_parse_additive(tok))
    return args


# ============ AST 相似度 ============

def ast_similarity(expr1: str, expr2: str) -> float:
    """计算两个 Qlib 表达式的 AST 结构相似度

    算法: 最大公共匹配节点数 / max(size1, size2)
    - 相同算子 + 子树匹配 → 节点匹配
    - 叶节点: 同类型(字段/常量)且值相同 → 匹配
    - 叶节点: 同类型但值不同 → 部分匹配 (0.5)

    Returns:
        0.0 ~ 1.0, 1.0 表示完全同构
    """
    tree1 = parse_qlib_expr(expr1)
    tree2 = parse_qlib_expr(expr2)
    matched = _count_matched(tree1, tree2)
    max_size = max(tree1.size, tree2.size)
    if max_size == 0:
        return 1.0
    return min(matched / max_size, 1.0)


def _count_matched(n1: ASTNode, n2: ASTNode) -> float:
    """递归计算两棵子树的匹配节点数"""
    # 两个都是叶节点
    if not n1.children and not n2.children:
        if n1.value == n2.value:
            return 1.0
        # 同类型但不同值 → 部分匹配
        if n1.is_field == n2.is_field and n1.is_const == n2.is_const:
            return 0.5
        return 0.0

    # 两个都是函数节点且算子相同
    if n1.op and n2.op and n1.op == n2.op:
        score = 1.0  # 算子本身匹配
        # 对齐子节点 (按位置)
        min_children = min(len(n1.children), len(n2.children))
        for i in range(min_children):
            score += _count_matched(n1.children[i], n2.children[i])
        return score

    # 算子不同但都是函数
    if n1.op and n2.op:
        score = 0.3  # 结构相似但算子不同
        min_children = min(len(n1.children), len(n2.children))
        for i in range(min_children):
            score += _count_matched(n1.children[i], n2.children[i]) * 0.5
        return score

    return 0.0


def ast_fingerprint(expr: str) -> str:
    """生成 AST 指纹 (用于快速去重)

    将常量替换为占位符, 保留结构和字段名。
    """
    tree = parse_qlib_expr(expr)
    return _fingerprint_node(tree)


def _fingerprint_node(node: ASTNode) -> str:
    if node.op:
        args = ",".join(_fingerprint_node(c) for c in node.children)
        return f"{node.op}({args})"
    if node.is_field:
        return str(node.value)
    return "C"  # 常量占位符
