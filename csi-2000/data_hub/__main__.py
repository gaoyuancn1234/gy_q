"""python -m data_hub status|validate|init|ingest|dump|refresh"""
from __future__ import annotations

import argparse
import sys

from data_hub.ingest import ingest_daily, init_layout
from data_hub.paths import TUSHARE_START
from data_hub.validate import report, validate


def _print_report(info: dict) -> None:
    """把核对摘要打到 stdout。"""
    keys = [
        "universe", "daily_dir", "daily_files", "daily_first",
        "daily_last", "qlib_dir", "qlib_days", "qlib_first",
        "qlib_last", "membership_snaps", "membership_union",
        "qlib_feature_dirs", "minute_dir", "minute_files",
        "sample_rows", "sample_ok",
    ]
    for k in keys:
        print(f"{k}: {info.get(k)}")
    if info.get("membership_error"):
        print(f"membership_error: {info['membership_error']}")


def main(argv: list[str] | None = None) -> int:
    """中台入口。默认 status, 不调接口。"""
    ap = argparse.ArgumentParser(description="Tushare 数据中台")
    ap.add_argument(
        "cmd",
        nargs="?",
        default="status",
        choices=["status", "validate", "init", "ingest",
                 "dump", "refresh"],
    )
    ap.add_argument("--universe", default="csi2000")
    ap.add_argument("--start", default=TUSHARE_START)
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--commit",
        action="store_true",
        help="dump/refresh 时替换生产 Qlib 目录",
    )
    args = ap.parse_args(argv)
    if args.cmd == "init":
        init_layout()
        _print_report(report(args.universe))
        return 0
    if args.cmd == "status":
        _print_report(report(args.universe))
        return 0
    if args.cmd == "validate":
        _print_report(report(args.universe))
        errors = validate(args.universe)
        if errors:
            print("FAIL")
            for e in errors:
                print(f"  - {e}")
            return 1
        print("PASS")
        return 0
    if args.cmd == "ingest":
        ingest_daily(args.universe, args.start, args.end)
        return 0
    if args.cmd == "dump":
        from data_hub.dump_qlib import dump_qlib
        dump_qlib(args.universe, commit=args.commit)
        return 0
    if args.cmd == "refresh":
        n = ingest_daily(args.universe, args.start, args.end)
        if n == 0 and not args.commit:
            print("[hub] 无新日线, 跳过 dump")
            return 0
        from data_hub.dump_qlib import dump_qlib
        dump_qlib(args.universe, commit=args.commit)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
