"""qlib 数据目录解析。换股票池只改 yaml, 不要各写一份 provider_uri。

csi300 历史目录名是 cn_data_bs; 其余是 cn_data_{universe}。
中证2000 只走 Tushare, 禁止新浪/BaoStock 覆盖供给层。
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


def parse_exec_lag(cfg: dict | None = None,
                   config_path: str | Path | None = None) -> int:
    """成交滞后天数。0 = 信号日 14:50 收盘竞价; 1 = T+1 收盘。

    2026-09-12 抽出。此前全树 9 处写 `int(cfg.get("exec_lag", 1) or 1)`,
    而 **`0 or 1` 求值成 1** —— yaml 明写 exec_lag: 0, 代码一律读成 1。

    后果:
      - paper_trader 的 8 相位验收全部跑成 T+1 收盘成交, 不是路线 B。
        标签是 Ref($open,-1)/$close-1 (T收盘->T+1开盘), 在 T+1 收盘买入
        等于整段隔夜跳已经错过 —— 验收数字描述的不是配置里那个策略。
      - daily auction 的 `if lag != 0: return 2` 永远成立, 周一生产入口
        被自己挡住, 还提示"先切 exec_lag=0"(而配置本来就是 0)。
      - paper_trader 打印的是原始值 0、执行用的是 1, 日志主动误导。

    `or` 对 0 失效是 Python 常见坑; 键存在且为 0 时必须走 None 判定。
    """
    if cfg is None:
        p = Path(config_path) if config_path else CONFIG_FILE
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    v = cfg.get("exec_lag", 1)
    if v is None:
        return 1
    return int(v)


def prediction_pkl(preset: str | None = None,
                   config_path: str | Path | None = None) -> Path:
    """当前配置对应的 rolling 预测 pkl 绝对路径。

    2026-09-12 新增。此前 5 个分析脚本各自写死
    `D_expand_3v_3r_alpha158_LightGBM.pkl` —— 而生产 preset 早已是
    `alpha158_ovn`, 该文件**根本不存在**, 脚本一跑就 FileNotFoundError。
    其中 `_trunc_ic.py` 正是路线 B 的杀/走判据, 它跑不起来意味着这道门
    从来没验过。

    文件名结构是 {rolling_config}_{preset}_{model}.pkl, 三段都在 yaml 里,
    写死任何一段都会随配置漂移。
    """
    pth = Path(config_path) if config_path else CONFIG_FILE
    with open(pth, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    name = (f"{cfg.get('rolling_config', 'D_expand_3v_3r')}"
            f"_{preset or cfg.get('preset', 'alpha158')}"
            f"_{cfg.get('model', 'LightGBM')}.pkl")
    return (PROJECT_DIR / cfg.get('model_cache_dir',
                                  'factor_lab/results/rolling/predictions')
            / name)
