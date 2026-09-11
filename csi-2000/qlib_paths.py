"""qlib 数据目录解析 —— 单一实现

为什么要有这个
--------------
provider_uri 此前在 3 处硬编码成 `cn_data_bs`(paper_trader / reconcile /
run_phase_test)，另有若干处各自从配置推导。后果是**改配置里的 instruments
不会真正换源**: 代码照样去读沪深300 的目录，而 qlib 对缺失成分股不报错、
返回全 NaN —— 换池变成一次沉默失败。

CLAUDE.md 记着 2026-09-02 切 CSI 800 当天回退，原因之一就是这个;
待办里也写着"若要重做，先解决 provider_uri 硬编码"。这里就是那件事。

目录命名沿用既有约定: csi300 的目录历史上叫 cn_data_bs(baostock 时代留下
的名字，数据源早已换成新浪，但目录名不动以免动到既有数据)，其余按
cn_data_{universe}。
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_DIR / "config" / "signal_config.yaml"
# 中证2000 rolling 训练窗从 2019-10 起; 刷新日线必须覆盖这段,
# 不能用 data_setup_tushare 默认的 2024-01-01, 否则训练集被砍掉.
TUSHARE_START = "2019-10-01"


def current_universe(config_path: str | Path | None = None) -> str:
    """从 signal_config.yaml 读当前股票池"""
    p = Path(config_path) if config_path else CONFIG_FILE
    with open(p, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("instruments", "csi300")


def qlib_dir(universe: str | None = None,
             config_path: str | Path | None = None) -> Path:
    """股票池 -> qlib 数据目录 (绝对路径)"""
    u = universe or current_universe(config_path)
    name = "cn_data_bs" if u == "csi300" else f"cn_data_{u}"
    return (Path("~/.qlib/qlib_data") / name).expanduser()


def qlib_provider_uri(universe: str | None = None,
                      config_path: str | Path | None = None) -> str:
    """给 qlib.init(provider_uri=...) 用的字符串"""
    return str(qlib_dir(universe, config_path))


def data_refresh_chain(universe: str | None = None) -> list[tuple[str, str]]:
    """股票池 -> 数据刷新模块链 [(标签, python -m 模块名), ...]

    中证2000 只能走 Tushare: 新浪/baostock 没有时点成分股, 且 DailyRunner
    用新浪全量重写会把已建好的 cn_data_csi2000 盖掉.
    """
    u = universe or current_universe()
    if u == "csi2000":
        # 只走 Tushare。新浪会盖掉时点成分股, 见 data_hub.guard。
        return [("Tushare", "qlib_engine.data_setup_tushare")]
    return [("新浪", "qlib_engine.data_setup_sina"),
            ("BaoStock", "qlib_engine.data_setup")]


def qlib_init_kwargs(universe: str | None = None,
                     config_path: str | Path | None = None) -> dict:
    """qlib.init 参数, 中证2000 限制 kernels 防止 MemoryError"""
    import yaml as _yaml
    p = Path(config_path) if config_path else CONFIG_FILE
    with open(p, encoding="utf-8") as f:
        cfg = _yaml.safe_load(f) or {}
    u = universe or cfg.get("instruments", "csi300")
    kw = {"provider_uri": qlib_provider_uri(u, p), "region": "cn"}
    kernels = cfg.get("qlib_kernels")
    if kernels is None and u != "csi300":
        kernels = 2
    if kernels:
        kw["kernels"] = int(kernels)
    return kw
