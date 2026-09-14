"""开源量价因子筛选。选参只看 2021-2023, 2024 以后不当选参。

用法:
    python -m factor_lab.screen_open_factors
"""
from __future__ import annotations

import json
from pathlib import Path

# 用户指定
UNIVERSE = "csi2000"
FIT_START = "2021-01-04"
FIT_END = "2022-12-30"
HOLD_START = "2023-01-03"
HOLD_END = "2023-12-29"
# |RankIC| 与 |ICIR| 门槛, 两段同号才留。
MIN_ABS_IC = 0.015
MIN_ABS_ICIR = 0.25
MAX_CORR = 0.70
MAX_KEEP = 12
OUT_NAME = "os_selected.json"

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "factor_lab" / "results" / "factor_eval"
import sys
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
import qlib_compat  # noqa: F401


def _ic_table(exprs, start, end):
    """一段区间的 RankIC / ICIR。"""
    from factor_lab.evaluation.single_factor import evaluate_with_qlib
    from factor_lab.factors.open_source import OVN_LABEL
    return evaluate_with_qlib(
        exprs,
        instruments=UNIVERSE,
        start_time=start,
        end_time=end,
        label_expr=OVN_LABEL,
    )


def _greedy_keep(names, factor_df, max_corr, max_keep):
    """按名单顺序贪心去相关。"""
    kept = []
    for name in names:
        if len(kept) >= max_keep:
            break
        if name not in factor_df.columns:
            continue
        ok = True
        for old in kept:
            pair = factor_df[[name, old]].dropna()
            if len(pair) < 200:
                continue
            c = abs(float(pair[name].corr(pair[old])))
            if c >= max_corr:
                ok = False
                break
        if ok:
            kept.append(name)
    return kept


def main() -> int:
    """筛选并写出 os_selected.json。"""
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    from qlib_paths import qlib_init_kwargs
    from factor_lab.factors.open_source import get_extra_exprs

    qlib.init(**qlib_init_kwargs(UNIVERSE))
    exprs = get_extra_exprs()
    print(f"候选 {len(exprs)}  拟合 {FIT_START}~{FIT_END}  "
          f"确认 {HOLD_START}~{HOLD_END}")
    fit = _ic_table(exprs, FIT_START, FIT_END)
    hold = _ic_table(exprs, HOLD_START, HOLD_END)
    fit = fit.set_index("factor")
    hold = hold.set_index("factor")
    passed = []
    for name in fit.index:
        if name not in hold.index:
            continue
        ic_f = float(fit.loc[name, "mean_IC"])
        ir_f = float(fit.loc[name, "ICIR"])
        ic_h = float(hold.loc[name, "mean_IC"])
        ir_h = float(hold.loc[name, "ICIR"])
        if abs(ic_f) < MIN_ABS_IC or abs(ir_f) < MIN_ABS_ICIR:
            continue
        if ic_f * ic_h <= 0:
            continue
        if abs(ic_h) < MIN_ABS_IC * 0.5:
            continue
        passed.append((name, abs(ir_f), ic_f, ir_f, ic_h, ir_h))
    passed.sort(key=lambda x: -x[1])
    print(f"两段同号且过门槛: {len(passed)}")
    for row in passed:
        print(f"  {row[0]:22} fitIC {row[2]:+.4f} "
              f"fitIR {row[3]:+.3f} holdIC {row[4]:+.4f}")

    names = [p[0] for p in passed]
    mapping = dict(exprs)
    fields = [mapping[n] for n in names]
    inst = D.instruments(UNIVERSE)
    factor_df = D.features(
        instruments=inst, fields=fields,
        start_time=FIT_START, end_time=HOLD_END,
    )
    factor_df.columns = names
    kept = _greedy_keep(names, factor_df, MAX_CORR, MAX_KEEP)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / OUT_NAME
    out.write_text(json.dumps(kept, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    fit.to_csv(OUT_DIR / "os_ic_fit.csv")
    hold.to_csv(OUT_DIR / "os_ic_hold.csv")
    print(f"保留 {len(kept)}: {kept}")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
