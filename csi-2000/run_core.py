#!/usr/bin/env python3
"""CSI2000 核心链路冒烟.

检查配置、qlib 日线、预测缓存、单日信号; 可选模拟盘回放.
不刷新数据、不重训、不动原 CSI300 目录.

用法:
    python run_core.py
    python run_core.py --replay
"""
import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

REPLAY_START = "2024-01-02"
REPLAY_END = "2026-09-09"


def _check_config():
    """读配置并核对本目录指向中证2000."""
    from qlib_paths import current_universe, qlib_dir, data_refresh_chain
    import yaml
    cfg_path = PROJECT_DIR / "config" / "signal_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    uni = current_universe()
    print(f"  目录: {PROJECT_DIR}")
    print(f"  股票池: {uni}")
    print(f"  preset: {cfg.get('preset')}")
    print(f"  topk={cfg.get('topk')}  adaptive={cfg.get('adaptive_strategy')}")
    print(f"  数据目录: {qlib_dir()}")
    print(f"  刷新链: {data_refresh_chain(uni)}")
    if uni != "csi2000":
        raise SystemExit("配置不是 csi2000, 停")
    preset = cfg.get("preset")
    if preset not in ("alpha158", "alpha158_ovn"):
        raise SystemExit(f"核心路径 preset 应为 alpha158 或 alpha158_ovn, 现为 {preset}")
    return cfg


def _check_qlib():
    """初始化 qlib, 确认成分股和收盘价能读."""
    import qlib
    from qlib.data import D
    from qlib_paths import qlib_init_kwargs
    kw = qlib_init_kwargs()
    print(f"  qlib.init {kw}")
    qlib.init(**kw)
    inst = D.list_instruments(
        instruments=D.instruments("csi2000"), as_list=True)
    print(f"  instruments(csi2000): {len(inst)} 只")
    if len(inst) < 1500:
        raise SystemExit(f"成分股过少: {len(inst)}")
    sample = inst[:3]
    px = D.features(sample, ["$close"],
                    start_time="2026-09-01", end_time="2026-09-09")
    n = 0 if px is None else len(px)
    print(f"  抽样 3 只 2026-09 收盘价: {n} 行")
    if n == 0:
        raise SystemExit("收盘价读出为空")
    cal = Path(kw["provider_uri"]) / "calendars" / "day.txt"
    last = cal.read_text(encoding="utf-8").strip().splitlines()[-1]
    print(f"  日历末日: {last}")
    return last


def _check_signal():
    """加载预测并生成最近一个交易日的 TopK."""
    from factor_lab.signal_generator import SignalGenerator
    sg = SignalGenerator()
    pred = sg.load_predictions()
    dates = pred.index.get_level_values(0)
    print(f"  预测: {len(pred)} 行  "
          f"{dates.min().date()} ~ {dates.max().date()}")
    sig = sg.get_signal()
    if sig.get("error"):
        raise SystemExit(sig["error"])
    print(f"  信号日 {sig['date']}  regime={sig['regime']}  "
          f"TopK={sig['effective_topk']}")
    print(f"  前5: {sig['target_stocks'][:5]}")
    if len(sig["target_stocks"]) < 8:
        raise SystemExit("信号名单过短")
    return sig


def _replay():
    """用已有预测跑模拟盘, 状态写在本目录 paper_trading."""
    from factor_lab.paper_trader import PaperTrader
    pt = PaperTrader()
    result = pt.replay(start_date=REPLAY_START, end_date=REPLAY_END)
    print(f"  总收益 {result.get('total_return')}")
    print(f"  Sharpe {result.get('sharpe')}")
    print(f"  最大回撤 {result.get('max_drawdown')}")
    return result


def main():
    """跑核心检查; --replay 再加模拟盘回放."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true",
                    help="额外跑 2024-01~2026-09 模拟盘")
    args = ap.parse_args()
    print("=== 1. 配置 ===")
    _check_config()
    print("=== 2. qlib 数据 ===")
    _check_qlib()
    print("=== 3. 信号 ===")
    _check_signal()
    print("核心链路可读.")
    if args.replay:
        print("=== 4. 模拟盘回放 ===")
        _replay()
    print("完成.")


if __name__ == "__main__":
    main()
