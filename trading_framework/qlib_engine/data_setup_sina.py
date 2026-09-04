#!/usr/bin/env python3
"""Qlib 数据下载 — 新浪源 (akshare)

data_setup.py 的备用数据源。2026-09-02 新增，起因是 baostock 的 10030 数据端口
持续拒绝服务 (连上即被对端关闭，客户端在死连接上空转、既不报错也不返回数据)。
同一台主机的 80/443 端口数据往返正常，所以不是网络或代理问题。

与 baostock 源的差异:
  - 速度: 约 1.1s/只 (baostock 约 6s/只)
  - 数据源: 新浪 (finance.sina.com.cn)，HTTP 协议，不走 10030 裸 TCP
  - **isST 不可得**: 新浪日线接口没有 ST 标记，该字段全部写 NaN。
    ST 过滤将失效 —— 与旧的 cn_data_bs 数据集状况相同 (见 CLAUDE.md)。
    下游依赖 `.notna().any()` 判定字段可用性，因此会被正确识别为"不可用"
    而非静默当成"没有 ST 股"。
  - turn: 新浪给的是比率 (0.001765)，baostock 是百分数 (0.1765)，已 ×100 对齐
  - pctChg: 新浪不直接提供，按 close 相对前收盘计算

成分股沿用 data/{universe}_membership.json 缓存 (baostock 之前采样好的时点快照)，
并复用 data_setup.snapshots_to_intervals —— 时点成分股逻辑只有一份实现。

用法:
    python -m qlib_engine.data_setup_sina
    python -m qlib_engine.data_setup_sina --limit 10        # 先跑通再全量
    python -m qlib_engine.data_setup_sina --target_dir ~/.qlib/qlib_data/cn_data_bs
"""

import json
import os
import shutil
import time
from pathlib import Path

# akshare 内部的 requests.get 全都没传 timeout —— 一次瞬时网络故障就能让下载
# 无限期挂住。2026-09-04 实测: 整个 564 只的下载卡在第 1 只上 8.5 小时，
# 只用了 13 秒 CPU(纯阻塞在 socket)，且进度条 20 只才打一次、还是回车覆盖，
# 日志里一个字都没有，外部完全看不出卡在哪。
# 超时护栏与实盘取价、交易日判断共用 net_guard，避免各写一份再分叉。
from net_guard import run_with_timeout as _with_timeout, install_default_request_timeout

REQUEST_TIMEOUT = 20          # 单个 HTTP 请求
PER_STOCK_TIMEOUT = 90        # 单只股票(含重试)的总预算
_TTY = __import__("sys").stdout.isatty()



import numpy as np
import pandas as pd

from .data_setup import (
    FIELDS, DEFAULT_TARGET_DIR,
    snapshots_to_intervals, _bao_to_qlib_instrument,
    _build_calendar, _build_instruments, _build_features,
)

INDEX_CODE = "sh000300"          # 新浪指数代码
INDEX_INSTRUMENT = "SH000300"


def _load_membership(universe: str) -> dict:
    """读取时点成分股缓存 (baostock 采样产物，本模块只读不写)"""
    cache_file = (Path(__file__).resolve().parent.parent / "data"
                  / f"{universe}_membership.json")
    if not cache_file.exists():
        raise RuntimeError(
            f"成分股缓存不存在: {cache_file}\n"
            f"本模块不查询指数成分 (新浪无历史成分接口)，依赖 baostock 采样好的缓存。"
        )
    with open(cache_file, "r", encoding="utf-8") as f:
        snapshots = {k: set(v) for k, v in json.load(f).items()}
    print(f"[sina] 成分股缓存: {len(snapshots)} 个时点 ({min(snapshots)} ~ {max(snapshots)})")
    return snapshots


def _bao_to_sina(bao_code: str) -> str:
    """sh.600000 -> sh600000"""
    return bao_code.replace(".", "").lower()


def _normalize(df: pd.DataFrame, start_date: str, end_date: str,
               is_index: bool = False) -> pd.DataFrame:
    """把新浪返回的列对齐到 FIELDS，并截取到指定区间

    FIELDS = [open, close, high, low, volume, amount, turn, pctChg, isST]

    区间截取放在转成 datetime 之后：新浪返回的 date 列有时是 datetime.date、
    有时是字符串，直接和 Timestamp 比较会抛 TypeError 或静默错判。
    """
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"])
    for col in ("open", "close", "high", "low", "volume"):
        out[col] = pd.to_numeric(df[col], errors="coerce")

    # amount: 指数接口不返回成交额
    out["amount"] = (pd.to_numeric(df["amount"], errors="coerce")
                     if "amount" in df.columns else np.nan)

    # turn: 新浪是比率，baostock 是百分数
    if "turnover" in df.columns:
        out["turn"] = pd.to_numeric(df["turnover"], errors="coerce") * 100.0
    else:
        out["turn"] = np.nan

    # pctChg: 新浪不提供，按前收盘计算 (百分数，与 baostock 对齐)
    out["pctChg"] = out["close"].pct_change() * 100.0

    # isST: 新浪日线无此标记。指数补 0 (无 ST 概念)，个股写 NaN 表示"不可得"。
    out["isST"] = 0.0 if is_index else np.nan

    out = out.sort_values("date").reset_index(drop=True)
    # pctChg 依赖前一日收盘，必须在截取区间【之前】算完，否则区间首日为 NaN
    out = out[(out["date"] >= pd.Timestamp(start_date))
              & (out["date"] <= pd.Timestamp(end_date))]
    # 过滤停牌 (与 baostock 源一致)
    out = out[(out["volume"] > 0) & (out["close"] > 0)]
    if out.empty:
        return None
    return out[["date"] + FIELDS].reset_index(drop=True)




def _download_stock(ak, bao_code: str, start_date: str, end_date: str):
    """下载单只股票日线 (前复权 + 复权因子)

    额外取一次原始价来算 factor = 前复权价 / 原始价。
    qlib 只有在 `$factor` 存在时才会应用 trade_unit(整手 100 股)；
    缺失时允许买零碎股，对 10 万本金的回测失真严重。
    多一次请求 (~1s/只)，换回测能真实反映整手约束。
    """
    sym = _bao_to_sina(bao_code)
    df = _with_timeout(lambda: ak.stock_zh_a_daily(symbol=sym, adjust="qfq"),
                       PER_STOCK_TIMEOUT, f"{sym} qfq")
    if df is None or df.empty:
        return None
    out = _normalize(df, start_date, end_date)
    if out is None:
        return None
    try:
        raw = _with_timeout(lambda: ak.stock_zh_a_daily(symbol=sym, adjust=""),
                            PER_STOCK_TIMEOUT, f"{sym} raw")
        if raw is not None and not raw.empty:
            raw = raw.copy()
            raw["date"] = pd.to_datetime(raw["date"])
            r = raw.set_index("date")["close"]
            f = out.set_index("date")["close"] / r.reindex(out["date"]).values
            out["factor"] = f.values
    except Exception:
        pass          # 取不到就不写 factor，qlib 会退回 adjusted_price 模式
    return out


def _download_index(ak, start_date: str, end_date: str):
    """下载沪深300指数日线"""
    df = ak.stock_zh_index_daily(symbol=INDEX_CODE)
    if df is None or df.empty:
        return None
    return _normalize(df, start_date, end_date, is_index=True)


def _swap_in(build_path, target_path, attempts: int = 12, wait: float = 10.0):
    """把 build_path 原子换成 target_path

    Windows 上重命名目录时，只要有别的进程正打开着目录里的文件就会
    PermissionError —— 22:00 的因子挖掘会连续 5 小时读这些 bin 文件，
    正好可能撞上。qlib 的读取是短连接式的，退避重试基本能等到空隙。

    全部重试失败也**不删** build_path: 里面是刚下完的完整数据集，
    等占用方退出后手工 rename 即可，不必再花几小时重下。
    """
    old_path = target_path.with_name(target_path.name + ".old")
    if old_path.exists():
        shutil.rmtree(old_path, ignore_errors=True)

    for i in range(1, attempts + 1):
        try:
            if target_path.exists():
                os.replace(target_path, old_path)   # 同盘改名，瞬时
            os.replace(build_path, target_path)
            shutil.rmtree(old_path, ignore_errors=True)
            print(f"[sina] 数据集已原子替换 (第 {i} 次尝试)")
            return
        except PermissionError as e:
            # 有可能 target 已经改名成功、build 那步失败，先把状态还原
            if old_path.exists() and not target_path.exists():
                try:
                    os.replace(old_path, target_path)
                except OSError:
                    pass
            if i == attempts:
                raise RuntimeError(
                    f"数据集替换失败({attempts} 次): {e} | "
                    f"完整数据已保留在 {build_path}，"
                    f"等占用进程退出后执行改名即可，无需重下。") from e
            print(f"[sina] 替换被占用，{wait:.0f}s 后重试 ({i}/{attempts}): {e}")
            time.sleep(wait)


def setup_qlib_data(target_dir: str = DEFAULT_TARGET_DIR,
                    start_date: str = "2018-01-01",
                    end_date: str = None,
                    universe: str = "csi300",
                    limit: int = 0,
                    retry: int = 3):
    """下载并构建 Qlib bin 数据 (新浪源)"""
    import akshare as ak

    t_total = time.time()
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    target_path = Path(target_dir).expanduser()

    # --limit 会写出残缺数据集，且构建阶段先 rmtree 目标目录。
    # 若指向生产目录，一次冒烟测试就会把可用数据集换成只有 N 只股票的残品，
    # 而下游回测不会报错、只会给出错误结论。所以强制要求换目录。
    if limit and target_path == Path(DEFAULT_TARGET_DIR).expanduser():
        raise SystemExit(
            f"--limit 是冒烟测试，会写出残缺数据集并覆盖 {target_path}。\n"
            f"请显式指定另一个目录，例如:\n"
            f"  --limit {limit} --target_dir ~/.qlib/qlib_data/_smoke_test")
    install_default_request_timeout(REQUEST_TIMEOUT)
    print(f"[sina] 目标: {target_path}")
    print(f"[sina] 区间: {start_date} ~ {end_date}")

    # 1. 时点成分股
    snapshots = _load_membership(universe)
    stocks, intervals = snapshots_to_intervals(snapshots)
    if limit:
        stocks = stocks[:limit]
        print(f"[sina] --limit {limit}: 只下载前 {len(stocks)} 只 (冒烟测试)")

    # 2. 逐只下载
    stock_map, all_dates, failed = {}, [], []
    total = len(stocks)
    t_dl = time.time()
    for i, bao_code in enumerate(stocks, 1):
        df = None
        for attempt in range(1, retry + 1):
            try:
                df = _download_stock(ak, bao_code, start_date, end_date)
                break
            except Exception as e:
                if attempt == retry:
                    failed.append((bao_code, f"{type(e).__name__}: {e}"))
                else:
                    time.sleep(1.5 * attempt)
        if df is not None and not df.empty:
            inst = _bao_to_qlib_instrument(bao_code)
            stock_map[inst] = df
            all_dates.extend(df["date"].tolist())
        # 进度输出: 交互式终端原地刷新; 重定向到日志文件时打整行并换行。
        # 原先只有回车覆盖那一种写法, 20 只才 flush 一次 —— 2026-09-04 下载
        # 卡在第 1 只上 8.5 小时, 日志里一个字都没有, 完全看不出卡在哪。
        # 行里带上 bao_code, 卡住时最后一行就指明了是哪只。
        if i % 20 == 0 or i == total:
            el = time.time() - t_dl
            eta = el / i * (total - i)
            line = (f"[sina] 下载中 [{i}/{total}] {bao_code} 成功 {len(stock_map)} "
                    f"失败 {len(failed)} | 已用 {el/60:.1f}min ETA {eta/60:.1f}min")
            if _TTY:
                print("\r" + line, end="", flush=True)
            else:
                print(line, flush=True)
    print()
    print(f"[sina] 成功 {len(stock_map)}/{total} 只 (耗时 {time.time()-t_dl:.0f}s)")

    if failed:
        print(f"[sina] 失败 {len(failed)} 只，前 5 个:")
        for code, err in failed[:5]:
            print(f"         {code}: {err[:80]}")

    # 下载成功率过低时显式失败，不要拿残缺数据去覆盖已有数据集
    if total and len(stock_map) / total < 0.9:
        raise RuntimeError(
            f"下载成功率过低 ({len(stock_map)}/{total})，已中止，未改动 {target_path}。"
            f"残缺数据集会让回测结果失真且难以察觉。")

    # 3. 指数
    print("[sina] 下载沪深300指数...")
    idx = _download_index(ak, start_date, end_date)
    if idx is None or idx.empty:
        raise RuntimeError("沪深300指数下载失败。指数是基准/择时依据，缺失必须显式失败。")
    stock_map[INDEX_INSTRUMENT] = idx
    all_dates.extend(idx["date"].tolist())
    print(f"[sina] 指数数据: {len(idx)} 条")

    # 4. 构建 Qlib 格式 (复用 data_setup 的构建函数)
    #
    # 2026-09-04: 改为"先建到旁边、再整体替换"。原先是直接 rmtree 生产目录
    # 再重建，中间有一段(数百个 bin 文件的写入时间)目录是空的或半截的。
    # qlib 对缺失数据不报错、返回全 NaN —— 任何在这个窗口里读数据的任务
    # (22:00 的因子挖掘、盘中监控、手动回测)都会拿到静默的垃圾结果。
    # 换成同盘旁路目录 + os.replace 原子替换，窗口缩到一次目录改名。
    build_path = target_path.with_name(target_path.name + ".building")
    if build_path.exists():
        shutil.rmtree(build_path)
    build_path.mkdir(parents=True)

    calendar = _build_calendar(all_dates, build_path)
    _build_instruments(stock_map, build_path, intervals, universe)
    _build_features(stock_map, calendar, build_path)

    _swap_in(build_path, target_path)

    print(f"\n[sina] 完成! provider_uri='{target_dir}' "
          f"(总耗时 {(time.time()-t_total)/60:.1f}min)")
    print("[sina] ⚠ isST 全部为 NaN (新浪源无此字段) —— ST 过滤在本数据集上不生效。")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Qlib A股数据下载（新浪 / akshare）')
    parser.add_argument('--target_dir', type=str, default=DEFAULT_TARGET_DIR)
    parser.add_argument('--start_date', type=str, default='2018-01-01')
    parser.add_argument('--end_date', type=str, default=None)
    parser.add_argument('--universe', default='csi300',
                        help='股票池，需存在对应的 data/{universe}_membership.json')
    parser.add_argument('--limit', type=int, default=0,
                        help='只下载前 N 只，用于冒烟测试 (会写出残缺数据集，勿用于回测)')
    args = parser.parse_args()
    setup_qlib_data(args.target_dir, args.start_date, args.end_date,
                    universe=args.universe, limit=args.limit)


if __name__ == '__main__':
    main()
