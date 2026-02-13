"""下载器基类 — 提供限速/重试/缓存/增量更新功能"""
import time
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


class BaseDownloader(ABC):
    """数据下载器基类

    子类只需实现:
    - source_name: 数据源名称
    - _fetch_one(code, start, end): 下载单只股票/单日数据
    """

    source_name: str = "base"

    def __init__(self, cache_dir: str = None, batch_size: int = 0,
                 batch_sleep: float = 0, retry: int = 3, retry_sleep: float = 5):
        """
        Args:
            cache_dir: CSV 缓存目录，默认 factor_lab/results/.cache/{source_name}
            batch_size: 每批下载多少只后暂停（0=不限）
            batch_sleep: 批间暂停秒数
            retry: 失败重试次数
            retry_sleep: 重试间隔秒数
        """
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent / "results" / ".cache" / self.source_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.batch_sleep = batch_sleep
        self.retry = retry
        self.retry_sleep = retry_sleep

    def _cache_path(self, key: str) -> Path:
        safe_key = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{key.replace('/', '_')}_{safe_key}.csv"

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else None)
                return df
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, df: pd.DataFrame):
        path = self._cache_path(key)
        df.to_csv(path, index=False)

    def _get_last_date(self, key: str) -> str | None:
        """获取缓存中的最后日期（用于增量更新）"""
        df = self._load_cache(key)
        if df is not None and "date" in df.columns and len(df) > 0:
            return pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")
        return None

    @abstractmethod
    def _fetch_one(self, code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """下载单只股票/单个条目的数据，子类实现"""
        ...

    def download(self, codes: list[str], start_date: str, end_date: str,
                 incremental: bool = True) -> dict[str, pd.DataFrame]:
        """批量下载，带限速/重试/增量更新

        Args:
            codes: 股票代码列表
            start_date: 起始日期
            end_date: 结束日期
            incremental: 是否增量更新（从缓存最后日期开始）

        Returns:
            {code: DataFrame} 字典
        """
        results = {}
        total = len(codes)

        for i, code in enumerate(codes):
            # 批间暂停
            if self.batch_size > 0 and i > 0 and i % self.batch_size == 0:
                print(f"  [{self.source_name}] 批间暂停 {self.batch_sleep}s... ({i}/{total})")
                time.sleep(self.batch_sleep)

            # 增量更新：读取缓存，确定起始日期
            actual_start = start_date
            cached_df = None
            if incremental:
                last_date = self._get_last_date(code)
                if last_date is not None:
                    next_day = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                    if next_day > end_date:
                        # 缓存已覆盖全部日期
                        cached_df = self._load_cache(code)
                        if cached_df is not None:
                            results[code] = cached_df
                            continue
                    actual_start = next_day
                    cached_df = self._load_cache(code)

            # 下载（带重试）
            df = None
            for attempt in range(self.retry):
                try:
                    df = self._fetch_one(code, actual_start, end_date)
                    break
                except Exception as e:
                    if attempt < self.retry - 1:
                        time.sleep(self.retry_sleep)
                    else:
                        print(f"  [{self.source_name}] {code} 下载失败: {e}")

            # 合并缓存
            if df is not None and len(df) > 0:
                if cached_df is not None:
                    df = pd.concat([cached_df, df], ignore_index=True)
                    if "date" in df.columns:
                        df = df.drop_duplicates(subset=["date"], keep="last")
                        df = df.sort_values("date").reset_index(drop=True)
                self._save_cache(code, df)
                results[code] = df
            elif cached_df is not None:
                results[code] = cached_df

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  [{self.source_name}] 进度 [{i+1}/{total}] 成功 {len(results)}")

        return results
