"""因子注册中心 — 管理所有因子的元数据和表达式"""
from dataclasses import dataclass, field
from enum import Enum


class FactorCategory(Enum):
    PRICE_VOLUME = "量价"
    VALUATION = "估值"
    FUNDAMENTAL = "基本面"
    MONEY_FLOW = "资金流"
    SENTIMENT = "情绪"
    TECHNICAL = "技术"


@dataclass
class FactorMeta:
    name: str                           # 因子名，如 "VWAP_RATIO"
    expr: str                           # Qlib 表达式，如 "($amount/$volume+1e-12)/$close"
    category: FactorCategory            # 分类
    description: str = ""               # 说明
    required_fields: list[str] = field(default_factory=list)  # 依赖的底层字段


class FactorRegistry:
    """全局因子注册中心"""

    def __init__(self):
        self._factors: dict[str, FactorMeta] = {}

    def register(self, meta: FactorMeta):
        """注册一个因子"""
        self._factors[meta.name] = meta

    def register_many(self, factors: list[FactorMeta]):
        """批量注册"""
        for f in factors:
            self.register(f)

    def get(self, name: str) -> FactorMeta | None:
        return self._factors.get(name)

    def get_all(self) -> list[FactorMeta]:
        return list(self._factors.values())

    def get_by_category(self, category: FactorCategory) -> list[FactorMeta]:
        return [f for f in self._factors.values() if f.category == category]

    def get_names(self, names: list[str] | None = None) -> list[str]:
        """获取因子名列表"""
        if names is None:
            return list(self._factors.keys())
        return [n for n in names if n in self._factors]

    def get_exprs(self, names: list[str] | None = None) -> list[str]:
        """获取因子表达式列表（可直接传给 QlibDataLoader）"""
        if names is None:
            return [f.expr for f in self._factors.values()]
        return [self._factors[n].expr for n in names if n in self._factors]

    def get_expr_pairs(self, names: list[str] | None = None) -> list[tuple[str, str]]:
        """获取 (name, expr) 对列表"""
        if names is None:
            return [(f.name, f.expr) for f in self._factors.values()]
        return [(n, self._factors[n].expr) for n in names if n in self._factors]

    def get_required_fields(self, names: list[str] | None = None) -> set[str]:
        """获取所有依赖的底层字段"""
        factors = self._factors.values() if names is None else [
            self._factors[n] for n in names if n in self._factors
        ]
        fields = set()
        for f in factors:
            fields.update(f.required_fields)
        return fields

    def filter_by_available_fields(self, available_fields: set[str]) -> list[FactorMeta]:
        """根据可用字段过滤因子"""
        result = []
        for f in self._factors.values():
            if all(rf in available_fields for rf in f.required_fields):
                result.append(f)
        return result

    def summary(self) -> str:
        """打印因子汇总"""
        lines = [f"因子注册中心: 共 {len(self._factors)} 个因子"]
        by_cat = {}
        for f in self._factors.values():
            cat_name = f.category.value
            by_cat.setdefault(cat_name, []).append(f.name)
        for cat, names in sorted(by_cat.items()):
            lines.append(f"  {cat}: {len(names)} 个")
        return "\n".join(lines)

    def __len__(self):
        return len(self._factors)

    def __contains__(self, name: str):
        return name in self._factors


# 全局单例
REGISTRY = FactorRegistry()


def register_factor(name: str, expr: str, category: FactorCategory,
                    description: str = "", required_fields: list[str] = None):
    """快捷注册函数"""
    REGISTRY.register(FactorMeta(
        name=name, expr=expr, category=category,
        description=description,
        required_fields=required_fields or [],
    ))
