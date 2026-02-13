#!/usr/bin/env python3
"""影子交易验证系统 — 候选策略对比验证

候选策略每日生成信号但不执行，与线上策略对比，
跑够验证期后人工决定切换。

用法:
    from shadow_manager import ShadowManager
    sm = ShadowManager()
    sid = sm.create_candidate(source="factor_miner:run_003",
                              config_overrides={"preset": "alpha158_val_mined"},
                              reason="VOL_REGIME_RATIO ICIR=0.58")
    sm.update_daily(sid, live_signal, shadow_signal, prices)
    sm.promote(sid)
"""
import gc
import json
import math
import shutil
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import yaml

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent
SHADOW_DIR = PROJECT_DIR / "shadow"
REGISTRY_FILE = SHADOW_DIR / "registry.json"
PRED_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling" / "predictions"


def _ensure_dirs():
    for d in [SHADOW_DIR, SHADOW_DIR / "configs",
              SHADOW_DIR / "state", SHADOW_DIR / "daily_log"]:
        d.mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding='utf-8'))
    return {}


def _save_registry(reg: dict):
    _ensure_dirs()
    REGISTRY_FILE.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding='utf-8')


def _next_shadow_id(reg: dict) -> str:
    nums = []
    for k in reg:
        if k.startswith("shadow_"):
            try:
                nums.append(int(k.split("_")[1]))
            except (ValueError, IndexError):
                continue
    if nums:
        return f"shadow_{max(nums) + 1:03d}"
    return "shadow_001"


class ShadowManager:
    """影子交易管理器"""

    def __init__(self):
        _ensure_dirs()
        self.registry = _load_registry()

    def create_candidate(self, source: str, config_overrides: dict,
                         reason: str, duration_days: int = 20) -> str:
        """创建 shadow candidate

        1. 分配 shadow_id
        2. 从 signal_config.yaml 拷贝 + 应用 overrides -> shadow config
        3. 写入 registry
        """
        shadow_id = _next_shadow_id(self.registry)

        # 拷贝 live config + 应用 overrides
        live_config_path = PROJECT_DIR / "config" / "signal_config.yaml"
        with open(live_config_path) as f:
            config = yaml.safe_load(f)
        config.update(config_overrides)

        # 独立 state 目录
        shadow_state_dir = SHADOW_DIR / "state" / shadow_id
        shadow_pred_dir = shadow_state_dir / "predictions"
        shadow_pred_dir.mkdir(parents=True, exist_ok=True)

        config['model_cache_dir'] = str(
            shadow_pred_dir.relative_to(PROJECT_DIR))
        config['quality_cache_dir'] = str(
            shadow_state_dir.relative_to(PROJECT_DIR))
        config['rolling_json_dir'] = str(
            shadow_state_dir.relative_to(PROJECT_DIR))

        # 保存 shadow config
        shadow_config_path = SHADOW_DIR / "configs" / f"{shadow_id}.yaml"
        with open(shadow_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        # 注册
        self.registry[shadow_id] = {
            "status": "active",
            "created_date": date.today().isoformat(),
            "source": source,
            "duration_days": duration_days,
            "elapsed_days": 0,
            "config_overrides": config_overrides,
            "reason": reason,
            "signal_config_path": str(
                shadow_config_path.relative_to(PROJECT_DIR)),
        }
        _save_registry(self.registry)

        print(f"  [shadow] 创建 {shadow_id}: {reason}")
        return shadow_id

    def retrain_for_shadow(self, shadow_id: str):
        """为 shadow 生成预测 (复用 retrain_pipeline 逻辑)"""
        cand = self.registry.get(shadow_id)
        if not cand:
            raise ValueError(f"Shadow {shadow_id} 不存在")

        shadow_config_path = PROJECT_DIR / cand['signal_config_path']
        with open(shadow_config_path) as f:
            config = yaml.safe_load(f)

        preset = config.get('preset', 'alpha158_val')
        rolling_config = config.get('rolling_config', 'D_expand_3v_3r')
        model_name = config.get('model', 'LightGBM')
        test_start = config.get('test_start', '2024-01-01')
        test_end = config.get('test_end', '2026-06-30')

        print(f"  [shadow] retrain {shadow_id}: preset={preset}")

        import multiprocessing
        multiprocessing.set_start_method('fork', force=True)

        import pickle
        import pandas as pd
        from factor_lab.run_rolling_benchmark import (
            generate_rolling_windows, build_model, ROLLING_CONFIGS,
        )

        rc = ROLLING_CONFIGS[rolling_config]
        all_windows = generate_rolling_windows(
            rolling_config, rc, test_start, test_end)

        import qlib
        from qlib.data.dataset import DatasetH
        try:
            qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region='cn')
        except Exception:
            pass

        preds = []
        window_details = []

        for w in all_windows:
            print(f"    Window {w['window_num']}: "
                  f"Pred [{w['pred_start']}, {w['pred_end']}]")
            t0 = time.time()

            if preset == 'alpha158':
                from qlib.contrib.data.handler import Alpha158
                handler = Alpha158(
                    start_time=w['train_start'], end_time=w['pred_end'],
                    fit_start_time=w['train_start'],
                    fit_end_time=w['train_end'],
                    instruments='csi300',
                )
            else:
                from factor_lab.factors.presets import build_handler
                handler = build_handler(
                    preset,
                    start_time=w['train_start'], end_time=w['pred_end'],
                    fit_start_time=w['train_start'],
                    fit_end_time=w['train_end'],
                )

            dataset = DatasetH(handler=handler, segments={
                "train": (w['train_start'], w['train_end']),
                "valid": (w['valid_start'], w['valid_end']),
                "test": (w['pred_start'], w['pred_end']),
            })

            model, fit_kwargs = build_model(model_name)
            model.fit(dataset, **fit_kwargs)
            pred = model.predict(dataset)

            if isinstance(pred.index, pd.MultiIndex):
                dates = pred.index.get_level_values(0)
                mask = ((dates >= pd.Timestamp(w['pred_start'])) &
                        (dates <= pd.Timestamp(w['pred_end'])))
                pred = pred[mask]

            best_iter = getattr(model.model, 'best_iteration', None)
            window_details.append({
                "window_num": w['window_num'],
                "pred_start": w['pred_start'],
                "pred_end": w['pred_end'],
                "n_samples": int(len(pred)),
                "best_iteration": best_iter,
                "train_time": round(time.time() - t0, 1),
            })
            preds.append(pred)
            print(f"      Best iter: {best_iter}, "
                  f"Samples: {len(pred)}, Time: {time.time()-t0:.1f}s")

            del handler, dataset, model, pred
            gc.collect()

        if not preds:
            print(f"  [shadow] {shadow_id}: 无预测结果")
            return

        combined = pd.concat(preds)
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep='last')]

        # 保存到 shadow 独立目录
        pred_dir = (SHADOW_DIR / "state" / shadow_id / "predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        pkl_name = f"{rolling_config}_{preset}_{model_name}.pkl"
        combined.to_pickle(pred_dir / pkl_name)

        # 保存 rolling info
        info = {
            "n_windows": len(window_details),
            "windows": window_details,
            "timestamp": datetime.now().isoformat(),
        }
        info_path = SHADOW_DIR / "state" / shadow_id / "rolling_info.json"
        info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

        # 生成 quality_score (SignalGenerator 需要)
        self._generate_quality_score(shadow_id, combined)

        print(f"  [shadow] {shadow_id} retrain 完成: "
              f"{len(combined)} samples")

    def _generate_quality_score(self, shadow_id: str, predictions):
        """生成 quality_score.pkl (SignalGenerator 依赖)"""
        try:
            import pickle
            import pandas as pd
            from qlib.data import D
            from factor_lab.run_signal_decay_benchmark import run_signal_analysis

            pred_dates = predictions.index.get_level_values(0)
            start = pred_dates.min().strftime('%Y-%m-%d')
            end = pred_dates.max().strftime('%Y-%m-%d')

            instruments = D.instruments('csi300')
            prices = D.features(instruments, ['$close'],
                                start_time=start, end_time=end)
            prices = prices.swaplevel().sort_index()
            close = prices['$close']

            valid_dates = close.index.get_level_values(0).unique()
            pred_valid = predictions[
                predictions.index.get_level_values(0).isin(valid_dates)]

            if len(pred_valid) == 0:
                print(f"    [quality] 无有效预测数据，跳过")
                return

            result = run_signal_analysis(pred_valid, close)

            quality_dir = SHADOW_DIR / "state" / shadow_id
            quality_dir.mkdir(parents=True, exist_ok=True)
            with open(quality_dir / "quality_score.pkl", 'wb') as f:
                pickle.dump(result['quality_score'], f)
            print(f"    [quality] 已生成 quality_score "
                  f"({len(result['quality_score'])} days)")
        except Exception as e:
            print(f"    [quality] 生成失败: {e}")

    def get_active_candidates(self) -> dict:
        """返回所有需要每日更新的 shadow (active + reverse_shadow)"""
        return {k: v for k, v in self.registry.items()
                if v.get('status') in ('active', 'reverse_shadow')}

    def update_daily(self, shadow_id: str, live_signal: dict,
                     shadow_signal: dict, prices: dict):
        """记录每日信号对比 + 计算理论日收益"""
        cand = self.registry.get(shadow_id)
        if not cand or cand['status'] not in ('active', 'reverse_shadow'):
            return

        today = date.today().isoformat()

        # 计算信号重叠率
        live_stocks = set(live_signal.get('target_stocks', []))
        shadow_stocks = set(shadow_signal.get('target_stocks', []))
        if live_stocks or shadow_stocks:
            overlap = len(live_stocks & shadow_stocks) / max(
                len(live_stocks | shadow_stocks), 1)
        else:
            overlap = 1.0

        # 理论日收益: 用昨日价格对比
        prev_prices = _load_prev_prices()
        shadow_return = calc_daily_returns(
            shadow_signal.get('target_stocks', []), prices, prev_prices)
        live_return = calc_daily_returns(
            live_signal.get('target_stocks', []), prices, prev_prices)
        # 保存今日价格供明天使用
        _save_prev_prices(prices)

        daily_entry = {
            "shadow_id": shadow_id,
            "target_stocks": shadow_signal.get('target_stocks', []),
            "regime": shadow_signal.get('regime', ''),
            "topk": shadow_signal.get('effective_topk', 0),
            "overlap_with_live": round(overlap, 4),
            "theoretical_return": round(shadow_return, 6),
            "live_return": round(live_return, 6),
        }

        # 写入每日日志
        log_file = SHADOW_DIR / "daily_log" / f"{today}.json"
        if log_file.exists():
            daily_log = json.loads(log_file.read_text(encoding='utf-8'))
        else:
            daily_log = {
                "date": today,
                "live": {
                    "target_stocks": live_signal.get('target_stocks', []),
                    "regime": live_signal.get('regime', ''),
                    "topk": live_signal.get('effective_topk', 0),
                },
                "candidates": {},
            }
        daily_log["candidates"][shadow_id] = daily_entry
        log_file.write_text(
            json.dumps(daily_log, ensure_ascii=False, indent=2),
            encoding='utf-8')

        # 更新 elapsed
        cand['elapsed_days'] = cand.get('elapsed_days', 0) + 1
        _save_registry(self.registry)

        # 更新汇总
        self._update_summary(shadow_id)

    def get_performance(self, shadow_id: str) -> dict:
        """获取累积绩效"""
        summary_file = SHADOW_DIR / "daily_log" / "summary.json"
        if not summary_file.exists():
            return {}
        summaries = json.loads(summary_file.read_text(encoding='utf-8'))
        return summaries.get(shadow_id, {})

    def promote(self, shadow_id: str):
        """晋升: 备份 live config -> 更新 preset -> 创建反转影子"""
        cand = self.registry.get(shadow_id)
        if not cand:
            raise ValueError(f"Shadow {shadow_id} 不存在")
        if cand.get('status') != 'active':
            raise ValueError(
                f"只能晋升 active shadow (当前: {cand.get('status')})")

        live_config_path = PROJECT_DIR / "config" / "signal_config.yaml"

        # 1. 备份
        bak_name = f"signal_config.yaml.bak_{date.today().strftime('%Y%m%d')}"
        bak_path = PROJECT_DIR / "config" / bak_name
        shutil.copy2(live_config_path, bak_path)
        print(f"  [shadow] 备份: {bak_path.name}")

        # 2. 更新 live config
        with open(live_config_path) as f:
            config = yaml.safe_load(f)
        config.update(cand.get('config_overrides', {}))
        with open(live_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"  [shadow] 已更新 signal_config.yaml")

        # 3. 标记 promoted
        cand['status'] = 'promoted'
        cand['promoted_date'] = date.today().isoformat()
        _save_registry(self.registry)

        # 4. 为旧基线创建反转影子
        reverse_id = self._create_reverse_shadow(shadow_id, bak_path)

        print(f"  [shadow] {shadow_id} 已晋升")
        if reverse_id:
            print(f"  [shadow] 旧基线 → {reverse_id} (反转验证 20 天)")
        print(f"  请运行 retrain_pipeline.py 使 live predictions 包含新因子")

    def _create_reverse_shadow(self, promoted_shadow_id: str,
                                bak_config_path: Path) -> str:
        """为旧基线创建反转影子 — 旧模型继续跑 20 天对比验证

        Args:
            promoted_shadow_id: 被晋升的 shadow ID
            bak_config_path: 旧基线 signal_config.yaml 备份路径
        Returns:
            反转 shadow ID, 或空字符串(失败时)
        """
        try:
            reverse_id = _next_shadow_id(self.registry)

            # 从备份读取旧基线 config
            with open(bak_config_path) as f:
                old_config = yaml.safe_load(f)

            # 独立 state 目录
            reverse_state_dir = SHADOW_DIR / "state" / reverse_id
            reverse_pred_dir = reverse_state_dir / "predictions"
            reverse_pred_dir.mkdir(parents=True, exist_ok=True)

            old_config['model_cache_dir'] = str(
                reverse_pred_dir.relative_to(PROJECT_DIR))
            old_config['quality_cache_dir'] = str(
                reverse_state_dir.relative_to(PROJECT_DIR))
            old_config['rolling_json_dir'] = str(
                reverse_state_dir.relative_to(PROJECT_DIR))

            # 保存 shadow config
            reverse_config_path = (SHADOW_DIR / "configs" /
                                   f"{reverse_id}.yaml")
            with open(reverse_config_path, 'w') as f:
                yaml.dump(old_config, f, default_flow_style=False,
                          allow_unicode=True)

            # 归档旧基线的 predictions + quality_score 到反转 shadow
            self._archive_baseline_data(reverse_state_dir, old_config)

            # 注册
            old_preset = old_config.get('preset', 'alpha158_val')
            self.registry[reverse_id] = {
                "status": "reverse_shadow",
                "created_date": date.today().isoformat(),
                "source": f"reverse:{promoted_shadow_id}",
                "duration_days": 20,
                "elapsed_days": 0,
                "config_overrides": {"preset": old_preset},
                "reason": f"旧基线反转验证 (preset={old_preset})",
                "signal_config_path": str(
                    reverse_config_path.relative_to(PROJECT_DIR)),
            }
            _save_registry(self.registry)

            print(f"  [shadow] 创建反转影子 {reverse_id}: "
                  f"旧基线 preset={old_preset}")
            return reverse_id
        except Exception as e:
            print(f"  [shadow] 创建反转影子失败: {e}")
            return ""

    def _archive_baseline_data(self, target_dir: Path,
                               old_config: dict):
        """将当前基线的 predictions + quality_score 拷贝到目标目录"""
        pred_dir = target_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

        # 从旧 config 构造 predictions 文件名
        rc = old_config.get('rolling_config', 'D_expand_3v_3r')
        preset = old_config.get('preset', 'alpha158_val')
        model = old_config.get('model', 'LightGBM')
        pkl_name = f"{rc}_{preset}_{model}.pkl"

        src_pred = PRED_DIR / pkl_name
        if src_pred.exists():
            shutil.copy2(src_pred, pred_dir / src_pred.name)

        # 拷贝 quality_score.pkl
        src_quality = (PROJECT_DIR / "factor_lab" / "results" /
                       "signal_decay" / "quality_score.pkl")
        if src_quality.exists():
            shutil.copy2(src_quality, target_dir / "quality_score.pkl")

    def archive(self, shadow_id: str):
        """封存模型 — 不再参与自动重训和日常信号生成"""
        cand = self.registry.get(shadow_id)
        if not cand:
            raise ValueError(f"Shadow {shadow_id} 不存在")
        if cand.get('status') != 'reverse_shadow':
            raise ValueError(
                f"只能封存反转 shadow (当前: {cand.get('status')})")
        cand['status'] = 'archived'
        cand['archived_date'] = date.today().isoformat()
        _save_registry(self.registry)
        print(f"  [shadow] {shadow_id} 已封存")

    def rollback(self, shadow_id: str):
        """回退到旧基线 — 反转实验证明旧模型更优"""
        cand = self.registry.get(shadow_id)
        if not cand:
            raise ValueError(f"Shadow {shadow_id} 不存在")
        if cand.get('status') != 'reverse_shadow':
            raise ValueError(
                f"只能回退反转 shadow (当前: {cand.get('status')})")

        # 1. 从 promote 备份恢复 signal_config.yaml (最准确)
        live_config_path = PROJECT_DIR / "config" / "signal_config.yaml"
        bak_files = sorted(
            (PROJECT_DIR / "config").glob("signal_config.yaml.bak_*"))
        if bak_files:
            shutil.copy2(bak_files[-1], live_config_path)
            print(f"  [shadow] 恢复 config: {bak_files[-1].name}")
        else:
            # fallback: 从 shadow config 恢复，替换路径为默认值
            shadow_config_path = PROJECT_DIR / cand['signal_config_path']
            with open(shadow_config_path) as f:
                old_config = yaml.safe_load(f)
            old_config['model_cache_dir'] = (
                'factor_lab/results/rolling/predictions')
            old_config['quality_cache_dir'] = (
                'factor_lab/results/signal_decay')
            old_config['rolling_json_dir'] = 'factor_lab/results/rolling'
            with open(live_config_path, 'w') as f:
                yaml.dump(old_config, f, default_flow_style=False,
                          allow_unicode=True)

        # 2. 从反转 shadow state 恢复 predictions + quality_score
        reverse_state = SHADOW_DIR / "state" / shadow_id
        reverse_pred = reverse_state / "predictions"

        for pkl_file in reverse_pred.glob("*.pkl"):
            dst = PRED_DIR / pkl_file.name
            shutil.copy2(pkl_file, dst)
            print(f"  [shadow] 恢复: {dst.name}")

        quality_src = reverse_state / "quality_score.pkl"
        quality_dst = (PROJECT_DIR / "factor_lab" / "results" /
                       "signal_decay" / "quality_score.pkl")
        if quality_src.exists():
            shutil.copy2(quality_src, quality_dst)
            print(f"  [shadow] 恢复: quality_score.pkl")

        # 3. 标记反转 shadow 为 rejected
        cand['status'] = 'rejected'
        cand['rejected_date'] = date.today().isoformat()
        _save_registry(self.registry)

        print(f"  [shadow] {shadow_id} 已回退 → 旧基线恢复")

    def reject(self, shadow_id: str):
        """拒绝 shadow"""
        cand = self.registry.get(shadow_id)
        if not cand:
            raise ValueError(f"Shadow {shadow_id} 不存在")
        cand['status'] = 'rejected'
        cand['rejected_date'] = date.today().isoformat()
        _save_registry(self.registry)
        print(f"  [shadow] {shadow_id} 已拒绝")

    def extend(self, shadow_id: str, extra_days: int):
        """延长验证期"""
        cand = self.registry.get(shadow_id)
        if not cand:
            raise ValueError(f"Shadow {shadow_id} 不存在")
        cand['duration_days'] = cand.get('duration_days', 20) + extra_days
        _save_registry(self.registry)
        print(f"  [shadow] {shadow_id} 延长 {extra_days} 天 "
              f"(总 {cand['duration_days']} 天)")

    def check_expired(self) -> list:
        """检查到期的 shadow，返回到期列表"""
        expired = []
        for sid, cand in self.registry.items():
            if cand['status'] not in ('active', 'reverse_shadow'):
                continue
            if cand.get('elapsed_days', 0) >= cand.get('duration_days', 20):
                expired.append(sid)
        return expired

    def get_status_text(self) -> str:
        """生成状态文本 (供飞书展示)"""
        active = self.get_active_candidates()
        archived = {k: v for k, v in self.registry.items()
                    if v.get('status') == 'archived'}

        if not active and not archived:
            return "当前无活跃影子验证"

        lines = []

        if active:
            # 分类: active vs reverse_shadow
            active_normal = {k: v for k, v in active.items()
                            if v.get('status') == 'active'}
            active_reverse = {k: v for k, v in active.items()
                             if v.get('status') == 'reverse_shadow'}

            if active_normal:
                lines.append(f"影子验证 ({len(active_normal)} 个):")
                for sid, cand in active_normal.items():
                    lines.extend(self._format_candidate(sid, cand))

            if active_reverse:
                lines.append(f"反转验证 ({len(active_reverse)} 个):")
                for sid, cand in active_reverse.items():
                    lines.extend(self._format_candidate(sid, cand))
                lines.append("  (反转期间自动重训暂停)")

        if archived:
            lines.append(f"已封存 ({len(archived)} 个):")
            for sid, cand in archived.items():
                lines.append(
                    f"  {sid}: {cand.get('reason', '')[:40]} "
                    f"({cand.get('archived_date', '?')})")

        return "\n".join(lines)

    def _format_candidate(self, sid: str, cand: dict) -> list:
        """格式化单个 candidate 展示"""
        elapsed = cand.get('elapsed_days', 0)
        duration = cand.get('duration_days', 20)
        perf = self.get_performance(sid)
        cum_ret = perf.get('cumulative_return', 0)
        avg_overlap = perf.get('avg_overlap', 0)

        has_data = perf.get('n_days', 0) > 0
        ret_str = f"{cum_ret:+.2%}" if has_data else "N/A"
        lines = [
            f"  {sid}: {elapsed}/{duration}天 "
            f"收益{ret_str} 重叠{avg_overlap:.0%}",
            f"    来源: {cand.get('source', '?')}",
            f"    原因: {cand.get('reason', '')[:50]}",
        ]
        return lines

    def get_expiry_report(self, shadow_id: str) -> str:
        """生成到期通知"""
        cand = self.registry.get(shadow_id)
        if not cand:
            return ""

        perf = self.get_performance(shadow_id)
        live_perf = self._get_live_performance(shadow_id)

        elapsed = cand.get('elapsed_days', 0)
        cum_ret = perf.get('cumulative_return', 0)
        sharpe = perf.get('sharpe', 0)
        mdd = perf.get('max_drawdown', 0)
        avg_overlap = perf.get('avg_overlap', 0)

        live_ret = live_perf.get('cumulative_return', 0)
        live_sharpe = live_perf.get('sharpe', 0)

        lines = [
            f"shadow 验证报告 - {shadow_id} ({elapsed}天)",
            f"来源: {cand.get('source', '?')}",
            f"理论收益: {cum_ret:+.1%} (vs live {live_ret:+.1%})",
            f"年化 Sharpe: {sharpe:.2f} (vs live {live_sharpe:.2f})",
            f"最大回撤: {mdd:.2%}",
            f"信号重叠: {avg_overlap:.0%}",
        ]

        is_reverse = cand.get('status') == 'reverse_shadow'

        if is_reverse:
            # 反转报告: shadow 是旧基线, live 是新模型
            if live_sharpe > sharpe and live_ret > cum_ret:
                lines.append("建议: 新模型确认更优，可封存旧基线")
            elif live_sharpe > sharpe:
                lines.append("建议: 新模型 Sharpe 更优，可封存")
            else:
                lines.append("建议: 旧基线表现更好，考虑回退")
        else:
            if sharpe > live_sharpe and cum_ret > live_ret:
                lines.append("建议: 可晋升")
            elif sharpe > live_sharpe:
                lines.append("建议: Sharpe 更优，但收益稍低，可延长观察")
            else:
                lines.append("建议: 表现不佳，可拒绝")

        lines.append("")
        lines.append("命令:")
        if is_reverse:
            lines.append(
                f"  python daily_runner.py --archive-shadow {shadow_id}")
            lines.append(
                f"  python daily_runner.py --rollback-shadow {shadow_id}")
        else:
            lines.append(
                f"  python daily_runner.py --promote-shadow {shadow_id}")
            lines.append(
                f"  python daily_runner.py --reject-shadow {shadow_id}")
        lines.append(
            f"  python daily_runner.py --extend-shadow {shadow_id} 10")

        return "\n".join(lines)

    def _update_summary(self, shadow_id: str):
        """更新累积汇总"""
        summary_file = SHADOW_DIR / "daily_log" / "summary.json"
        if summary_file.exists():
            summaries = json.loads(summary_file.read_text(encoding='utf-8'))
        else:
            summaries = {}

        # 收集该 shadow 所有每日数据
        daily_returns = []
        live_returns = []
        overlaps = []

        for log_file in sorted(
                (SHADOW_DIR / "daily_log").glob("20*.json")):
            daily = json.loads(log_file.read_text(encoding='utf-8'))
            entry = daily.get('candidates', {}).get(shadow_id)
            if entry:
                daily_returns.append(entry.get('theoretical_return', 0))
                live_returns.append(entry.get('live_return', 0))
                overlaps.append(entry.get('overlap_with_live', 0))

        if not daily_returns:
            return

        # 累积收益 (日收益累乘)
        cum = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in daily_returns:
            cum *= (1 + r)
            peak = max(peak, cum)
            dd = (cum - peak) / peak if peak > 0 else 0
            max_dd = min(max_dd, dd)

        cum_return = cum - 1.0
        n = len(daily_returns)

        # Sharpe
        if n >= 2:
            mean_r = sum(daily_returns) / n
            var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (n - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-9
            sharpe = (mean_r / std_r) * math.sqrt(252)
        else:
            sharpe = 0.0

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0

        summaries[shadow_id] = {
            "cumulative_return": round(cum_return, 6),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 6),
            "avg_overlap": round(avg_overlap, 4),
            "n_days": n,
        }
        summary_file.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding='utf-8')

    def _get_live_performance(self, shadow_id: str) -> dict:
        """获取同期 live 的绩效 (从 daily_log 中提取)"""
        daily_returns = []

        for log_file in sorted(
                (SHADOW_DIR / "daily_log").glob("20*.json")):
            daily = json.loads(log_file.read_text(encoding='utf-8'))
            entry = daily.get('candidates', {}).get(shadow_id)
            if entry:
                daily_returns.append(entry.get('live_return', 0))

        if not daily_returns:
            return {"cumulative_return": 0, "sharpe": 0}

        cum = 1.0
        for r in daily_returns:
            cum *= (1 + r)
        cum_return = cum - 1.0
        n = len(daily_returns)

        if n >= 2:
            mean_r = sum(daily_returns) / n
            var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (n - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-9
            sharpe = (mean_r / std_r) * math.sqrt(252)
        else:
            sharpe = 0.0

        return {
            "cumulative_return": round(cum_return, 6),
            "sharpe": round(sharpe, 3),
        }


PREV_PRICES_FILE = SHADOW_DIR / "prev_prices.json"


def _load_prev_prices() -> dict:
    """加载昨日价格"""
    if PREV_PRICES_FILE.exists():
        return json.loads(PREV_PRICES_FILE.read_text(encoding='utf-8'))
    return {}


def _save_prev_prices(prices: dict):
    """保存今日价格供明天计算日收益"""
    _ensure_dirs()
    PREV_PRICES_FILE.write_text(
        json.dumps(prices, ensure_ascii=False), encoding='utf-8')


def calc_daily_returns(target_stocks: list,
                       prices_today: dict,
                       prices_yesterday: dict) -> float:
    """计算 TopK 等权理论日收益 (外部使用)"""
    if not target_stocks:
        return 0.0
    returns = []
    for c in target_stocks:
        p1 = prices_today.get(c)
        p0 = prices_yesterday.get(c)
        if p1 and p0 and p0 > 0:
            returns.append(p1 / p0 - 1)
    return sum(returns) / len(returns) if returns else 0.0
