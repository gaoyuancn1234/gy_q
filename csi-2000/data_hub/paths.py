"""数据中台路径。用户指定的根目录和既有缓存都放在文件开头。"""
from __future__ import annotations

from pathlib import Path

# 中台根目录。原始 parquet 和校验产物都在这, Qlib bin 仍写到 SERVE。
HUB_ROOT = Path(r"C:\Users\1\.qlib\data_hub")
# 模型实际读取的 Qlib 供给层, 不要和原始层混写。
SERVE_ROOT = Path(r"C:\Users\1\.qlib\qlib_data")
PROJECT_DIR = Path(__file__).resolve().parent.parent
# 日线已在原树缓存, 先挂接, 不要复制 1683 个文件。
LEGACY_DAILY = (
    Path(r"c:\Users\1\Desktop\gy_Q\gy_q\trading_framework")
    / "factor_lab" / "results" / ".cache" / "tushare_csi2000"
)
# 分钟下载正在写入这里, 只挂接, 禁止挪走或双写。
LIVE_MINUTE = (
    Path(r"c:\Users\1\Desktop\gy_Q\gy_q\trading_framework")
    / "factor_lab" / "results" / ".cache" / "minute_csi2000"
)
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
