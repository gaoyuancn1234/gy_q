"""数据中台路径。用户指定的根目录和既有缓存都放在文件开头。"""
from __future__ import annotations

from pathlib import Path

# 中台根目录。原始 parquet 和校验产物都在这, Qlib bin 仍写到 SERVE。
# 本机用户目录, 不再写死另一台机器的 C:\Users\1。
HUB_ROOT = Path.home() / ".qlib" / "data_hub"
# 模型实际读取的 Qlib 供给层, 不要和原始层混写。
SERVE_ROOT = Path.home() / ".qlib" / "qlib_data"
PROJECT_DIR = Path(__file__).resolve().parent.parent
_CACHE = PROJECT_DIR / "factor_lab" / "results" / ".cache"
# 日线 parquet 缓存。没有旧树时 ingest / data_setup 会直接写入中台。
LEGACY_DAILY = _CACHE / "tushare_csi2000"
# 分钟缓存。没有旧树时保持空目录, 禁止挪走或双写。
LIVE_MINUTE = _CACHE / "minute_csi2000"
# 这些股票池禁止新浪/BaoStock 写入对应 Qlib 目录。
TUSHARE_ONLY = ("csi2000",)
TUSHARE_START = "2019-10-01"
UNIVERSE_INDEX = {
    "csi2000": "932000.CSI",
    "csi1000": "000852.SH",
    "csi500": "000905.SH",
    "csi300": "000300.SH",
}


def raw_dir() -> Path:
    """Tushare 原始层根目录。"""
    return HUB_ROOT / "raw" / "tushare"


def daily_raw_dir() -> Path:
    """按日 parquet。优先中台, 否则回落到已有缓存。"""
    hub = raw_dir() / "daily"
    if hub.exists() and any(hub.glob("*.parquet")):
        return hub
    if LEGACY_DAILY.exists():
        return LEGACY_DAILY
    return hub


def minute_raw_dir() -> Path:
    """15 分钟 parquet。下载未完成前指向正在写入的目录。"""
    hub = raw_dir() / "stk_mins"
    if hub.exists() and any(hub.glob("*.parquet")):
        return hub
    if LIVE_MINUTE.exists():
        return LIVE_MINUTE
    return hub


def trunc_daily_dir() -> Path:
    """14:45 截断日线 parquet。"""
    return raw_dir() / "daily_trunc_1445"


def membership_file(universe: str) -> Path:
    """时点成分股 json。"""
    return PROJECT_DIR / "data" / f"{universe}_membership.json"


def qlib_serve_dir(universe: str) -> Path:
    """Qlib bin 供给目录。csi300 历史名是 cn_data_bs。"""
    name = "cn_data_bs" if universe == "csi300" else f"cn_data_{universe}"
    return SERVE_ROOT / name


def calendar_file(universe: str) -> Path:
    """Qlib 交易日历。"""
    return qlib_serve_dir(universe) / "calendars" / "day.txt"


# 共享目录的归属标记 —— 2026-09-13 新增。
#
# raw/tushare/daily 与 raw/tushare/daily_trunc_1445 **不按股票池分目录**,
# 两棵树(csi-2000 / trading_framework)如果都往里写就会互相覆盖, 而且是
# 静默的: 文件名只有日期, 看不出是谁写的。
#
# 现状核实: CSI300 树完全不引用 data_hub(它走 data_setup_sina / BaoStock ->
# cn_data_bs), 所以当前没有冲突。但这是"恰好没人用"而非设计保证 ——
# 哪天 CSI300 树也启用 Tushare 就会踩上。
#
# 放一个 .owner 标记, 谁写谁认领; 发现不是自己的就明确报错, 不要静默覆盖。

def claim_shared_dir(d, owner: str = "csi-2000") -> None:
    """在共享目录放归属标记。归属不符时抛错, 而不是静默共用。"""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    f = d / ".owner"
    if f.exists():
        cur = f.read_text(encoding="utf-8").strip()
        if cur and cur != owner:
            raise RuntimeError(
                f"共享目录 {d} 归属 {cur!r}, 当前进程是 {owner!r}。\n"
                f"  这两个目录不按股票池分开, 交叉写入会静默覆盖对方的数据。\n"
                f"  要么给新树单独的 data_hub 根, 要么确认口径一致后删除 "
                f"{f} 重新认领。")
        return
    f.write_text(owner, encoding="utf-8")
