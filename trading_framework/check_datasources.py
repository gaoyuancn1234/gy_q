"""数据源健康检查 —— 实测各降级路径，而非凭代码推断

背景 (2026-09-05): baostock 因连续 45 分钟高频查询被拉黑
(error_code=10001011)，约 24 小时后自动解除。期间需要回答的问题是
"日常运行会不会受影响"，而这个问题不能靠读代码回答 ——
CLAUDE.md 记过四处"代码看起来有兜底、实际没有"的先例:
第三方库的网络调用没有超时，源一挂就永久阻塞，**挂起不抛异常**，
调用方写好的 except 降级分支根本轮不到执行。

所以这里逐条实跑: 交易日判断 / 实盘取价 / 数据源模块 / 信号生成。
每项都带耗时，卡住会直接看出来。

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
        print(f"{'✓' if ok else '✗'} {name:<26} {el:5.1f}s  {detail}")
        return ok
    except Exception as e:
        el = time.time() - t0
        print(f"✗ {name:<26} {el:5.1f}s  {type(e).__name__}: {str(e)[:70]}")
        return False


def main():
    results = []

    # 1. baostock 本身 —— 确认确实不可用
    def _bs():
        import baostock as bs
        from net_guard import run_with_timeout
        lg = run_with_timeout(bs.login, 30, 'login')
        return lg.error_code == '0', f"error_code={lg.error_code} {lg.error_msg}"
    bs_ok = check("baostock 登录", _bs)

    # 2. 交易日判断 —— 曾因 baostock 无超时卡死 1.5 小时
    def _cal():
        from market_calendar import is_trading_day
        from datetime import date
        r = is_trading_day(date(2026, 9, 4))     # 周五，交易日
        r2 = is_trading_day(date(2026, 9, 6))    # 周日
        return (r is True and r2 is False), f"9/4(五)={r}  9/6(日)={r2}"
    results.append(check("交易日判断", _cal))

    # 3. 实盘取价 —— 三级降级，新浪是主源
    def _px():
        from portfolio.live_portfolio import get_current_prices
        codes = ['SH600036', 'SH601318', 'SZ000001', 'SH600519', 'SZ300750']
        prices, from_cache = get_current_prices(codes)
        n = sum(1 for c in codes if prices.get(c, 0) > 0)
        src = "本地缓存" if from_cache else "实时源(新浪)"
        return n >= 4, f"{n}/{len(codes)} 只取到价  来源={src}"
    results.append(check("实盘取价", _px))

    # 4. 行情数据刷新 —— 新浪源能否单独工作
    def _sina():
        import importlib
        m = importlib.import_module('qlib_engine.data_setup_sina')
        has = hasattr(m, 'main') or hasattr(m, 'run')
        return has, f"data_setup_sina 可导入，入口={'有' if has else '无'}"
    results.append(check("新浪数据源模块", _sina))

    # 5. 信号生成 —— 只读本地缓存，不该依赖任何网络
    def _sig():
        from factor_lab.signal_generator import SignalGenerator
        sg = SignalGenerator()
        s = sg.get_signal()
        if 'error' in s:
            return False, s['error']
        h = s.get('health') or {}
        return True, (f"{s['date']} TopK={s['effective_topk']} "
                      f"健康={'通过' if h.get('ok', True) else '未通过'}")
    results.append(check("ML 信号生成", _sig))

    print()
    print(f"baostock: {'可用' if bs_ok else '不可用(黑名单)'}")
    print(f"降级路径: {sum(results)}/{len(results)} 项通过")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
