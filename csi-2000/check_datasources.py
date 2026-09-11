"""数据源健康检查 —— 实测 CSI2000 的 Tushare 路径

用法: python check_datasources.py     (退出码 0=全通过, 1=有失败)
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import qlib_compat  # noqa: F401


def check(name, fn, timeout=60):
    t0 = time.time()
    try:
        ok, detail = fn()
        el = time.time() - t0
        print(f"{'OK' if ok else 'X'} {name:<26} {el:5.1f}s  {detail}")
        return ok
    except Exception as e:
        el = time.time() - t0
        print(f"X {name:<26} {el:5.1f}s  {type(e).__name__}: {str(e)[:70]}")
        return False


def main():
    results = []

    def _cal():
        from market_calendar import is_trading_day
        from datetime import date
        r = is_trading_day(date(2026, 9, 4))
        r2 = is_trading_day(date(2026, 9, 6))
        return (r is True and r2 is False), f"9/4={r}  9/6={r2}"
    results.append(check("交易日判断", _cal))

    def _px():
        from portfolio.live_portfolio import get_current_prices
        codes = ["SH600036", "SZ000001", "SZ300750"]
        prices, from_cache = get_current_prices(codes)
        n = sum(1 for c in codes if prices.get(c, 0) > 0)
        src = "缓存" if from_cache else "Tushare"
        return n >= 2, f"{n}/{len(codes)} 只  来源={src}"
    results.append(check("Tushare 取价", _px))

    def _hub():
        from data_hub.validate import validate
        errs = validate("csi2000")
        return not errs, "PASS" if not errs else "; ".join(errs[:3])
    results.append(check("中台核对", _hub))

    print()
    print(f"Tushare 路径: {sum(results)}/{len(results)} 项通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
