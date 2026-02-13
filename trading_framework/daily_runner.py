#!/usr/bin/env python3
"""每日收盘后: 刷新数据 → 生成信号 → 推送飞书

Cron/launchd: 工作日 18:00

用法:
    python daily_runner.py              # 正常运行
    python daily_runner.py --dry-run    # 只生成不推送
    python daily_runner.py --morning-report  # 早间反思汇报
    python daily_runner.py --force      # 忽略交易日检查
    python daily_runner.py --shadow-status             # 查看影子验证状态
    python daily_runner.py --promote-shadow shadow_001  # 晋升 (自动创建反转影子)
    python daily_runner.py --reject-shadow shadow_001   # 拒绝
    python daily_runner.py --extend-shadow shadow_001 10  # 延长验证
    python daily_runner.py --archive-shadow shadow_002  # 封存 (确认新模型更优)
    python daily_runner.py --rollback-shadow shadow_002 # 回退到旧基线
    python daily_runner.py --experiment-status           # 查看实验盘状态
    python daily_runner.py --create-sentiment-experiment # 创建情绪哨兵实验
    python daily_runner.py --reject-experiment exp_001   # 终止实验
    python daily_runner.py --extend-experiment exp_001 15 # 延长实验
"""
import os
import sys
import json
import time
import shutil
import logging
import pickle
from datetime import datetime, date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(PROJECT_DIR / ".env", override=True)

# 日志
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "daily_runner.log", encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ============ 交易日检查 ============

def is_trading_day() -> bool:
    """检查今天是否为 A 股交易日 (周一~周五, 非节假日)"""
    today = date.today()
    # 周末直接跳过
    if today.weekday() >= 5:
        return False
    # 简单节假日检查: 用 BaoStock 日历判断
    try:
        import baostock as bs
        lg = bs.login()
        rs = bs.query_trade_dates(
            start_date=today.strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
        )
        while rs.next():
            row = rs.get_row_data()
            if row[1] == '1':  # is_trading_day
                bs.logout()
                return True
        bs.logout()
        return False
    except Exception as e:
        log.warning(f"交易日检查失败: {e}, 按工作日处理")
        return today.weekday() < 5


# ============ 数据刷新 ============

def refresh_daily_data() -> bool:
    """增量刷新 BaoStock 数据"""
    log.info("刷新 BaoStock 数据...")
    try:
        from qlib_engine.data_setup import setup_qlib_data
        t0 = time.time()
        setup_qlib_data()
        log.info(f"数据刷新完成 ({time.time() - t0:.0f}s)")
        return True
    except Exception as e:
        log.error(f"数据刷新失败: {e}")
        return False


# ============ 每日增量预测 ============

def _update_daily_predictions() -> bool:
    """检查并增量扩展当前 window 的预测

    数据刷新后，调用 retrain_pipeline.extend_rolling_predictions()
    在子进程中扩展预测到最新数据日期（避免 qlib.init 冲突）。

    Returns:
        True if predictions were updated
    """
    import pandas as pd
    import subprocess

    if not PRED_PKL.exists():
        log.warning("无已有预测文件，跳过增量预测")
        return False

    try:
        pred = pd.read_pickle(PRED_PKL)
        pred_last = pred.index.get_level_values(0).max()

        # 读取 test_end 配置
        import yaml
        config_path = PROJECT_DIR / "config" / "signal_config.yaml"
        with open(config_path) as f:
            sig_cfg = yaml.safe_load(f)
        test_end = sig_cfg.get('test_end', '2026-06-30')

        # 用子进程调用 extend_rolling_predictions (避免 qlib.init 冲突)
        log.info(f"当前预测最新日期: {pred_last.strftime('%Y-%m-%d')}，检查增量预测...")
        result = subprocess.run(
            [sys.executable, '-c',
             'import multiprocessing; '
             'multiprocessing.set_start_method("fork", force=True); '
             'import sys; sys.path.insert(0, "."); '
             f'from retrain_pipeline import extend_rolling_predictions; '
             f'pred = extend_rolling_predictions("{test_end}"); '
             f'last = pred.index.get_level_values(0).max().strftime("%Y-%m-%d"); '
             f'print(f"RESULT:{{last}}:{{len(pred)}}")'],
            capture_output=True, text=True, timeout=600,
            cwd=str(PROJECT_DIR),
        )

        if result.returncode != 0:
            stderr = result.stderr[-500:] if result.stderr else ''
            if 'RecorderInitializationError' not in stderr:
                log.error(f"增量预测子进程失败: {stderr}")
            return False

        output = result.stdout
        if 'RESULT:' in output:
            # 解析子进程输出
            for line in output.split('\n'):
                if line.startswith('RESULT:'):
                    parts = line.split(':')
                    new_last_str = parts[1]
                    new_last = pd.Timestamp(new_last_str)
                    if new_last > pred_last:
                        log.info(f"增量预测完成: "
                                 f"{pred_last.strftime('%Y-%m-%d')} → "
                                 f"{new_last.strftime('%Y-%m-%d')}")
                        return True
                    else:
                        log.info("预测已覆盖所有数据日期，无需更新")
                        return False
        elif '预测已是最新' in output:
            log.info("预测已是最新，无需更新")
            return False
        else:
            log.warning(f"增量预测输出异常: {output[-300:]}")
            return False

    except subprocess.TimeoutExpired:
        log.error("增量预测超时 (>10分钟)")
        return False
    except Exception as e:
        log.error(f"增量预测失败: {e}", exc_info=True)
        return False


# ============ 自动重训 ============

PRED_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling" / "predictions"
PRED_PKL = PRED_DIR / "D_expand_3v_3r_alpha158_val_LightGBM.pkl"


def _backup_predictions() -> 'pd.Series | None':
    """备份当前 predictions.pkl，返回旧预测数据"""
    if not PRED_PKL.exists():
        return None
    try:
        import pandas as pd
        old_pred = pd.read_pickle(PRED_PKL)
        backup_name = f"{PRED_PKL.stem}.bak_{date.today().strftime('%Y%m%d')}{PRED_PKL.suffix}"
        backup_path = PRED_PKL.parent / backup_name
        shutil.copy2(PRED_PKL, backup_path)
        log.info(f"预测备份: {backup_path.name}")
        return old_pred
    except Exception as e:
        log.error(f"备份预测失败: {e}")
        return None


def _rollback_predictions(old_pred: 'pd.Series'):
    """回退预测到备份版本"""
    try:
        old_pred.to_pickle(PRED_PKL)
        log.info("预测已回退到备份版本")
    except Exception as e:
        log.error(f"回退预测失败: {e}")


def _check_prediction_safety(old_pred: 'pd.Series', new_pred: 'pd.Series',
                              topk: int = 20) -> tuple:
    """对比最近30天共有日期的 TopK 重叠率

    Returns:
        (is_safe: bool, message: str)
    """
    import pandas as pd

    old_dates = old_pred.index.get_level_values(0).unique()
    new_dates = new_pred.index.get_level_values(0).unique()
    common_dates = old_dates.intersection(new_dates).sort_values()

    if len(common_dates) == 0:
        return True, "无共有日期，跳过安全检查"

    # 取最近30天共有日期
    recent = common_dates[-30:]
    overlaps = []

    for dt in recent:
        try:
            old_day = old_pred.loc[dt].nlargest(topk).index.tolist()
            new_day = new_pred.loc[dt].nlargest(topk).index.tolist()
            overlap = len(set(old_day) & set(new_day)) / topk
            overlaps.append(overlap)
        except Exception:
            continue

    if not overlaps:
        return True, "无法计算重叠率，跳过安全检查"

    avg_overlap = sum(overlaps) / len(overlaps)
    threshold = 0.5

    if avg_overlap >= threshold:
        return True, f"TopK重叠率: {avg_overlap:.0%} ≥ {threshold:.0%} → 安全"
    else:
        return False, f"TopK重叠率: {avg_overlap:.0%} < {threshold:.0%} → 突变"


def _should_skip_auto_retrain() -> tuple:
    """检查是否应跳过自动重训

    跳过条件: 存在 active reverse_shadow (模型刚切换，等反转实验结束)

    Returns:
        (should_skip: bool, reason: str)
    """
    try:
        from shadow_manager import ShadowManager
        sm = ShadowManager()
        for sid, cand in sm.registry.items():
            if cand.get('status') == 'reverse_shadow':
                elapsed = cand.get('elapsed_days', 0)
                duration = cand.get('duration_days', 20)
                return True, (f"跳过: 反转实验 {sid} 进行中 "
                              f"({elapsed}/{duration}天)")
        return False, ""
    except Exception as e:
        log.warning(f"反转实验检查失败: {e}")
        return False, ""


def _check_auto_retrain(dry_run: bool = False) -> bool:
    """自动重训检查

    触发条件: check_model_freshness().days_since > 60
    (距 pred_end 超过2个月，提前1个月重训留缓冲)

    Returns:
        True if retrain was performed (success or rollback)
    """
    # 反转实验进行中 → 跳过自动重训
    skip, reason = _should_skip_auto_retrain()
    if skip:
        log.info(f"自动重训: {reason}")
        return False

    try:
        from factor_lab.signal_generator import SignalGenerator
        sg = SignalGenerator()
        freshness = sg.check_model_freshness()

        if freshness['days_since'] <= 60:
            log.info(f"模型新鲜度: {freshness['message']}，无需重训")
            return False

        log.info(f"模型需要重训: {freshness['message']}")

        # 1. 备份旧预测
        old_pred = _backup_predictions()

        # 2. 执行重训
        from dateutil.relativedelta import relativedelta
        new_test_end = (date.today() + relativedelta(months=5)).strftime('%Y-%m-%d')
        log.info(f"自动重训: 目标 test_end = {new_test_end}")

        from retrain_pipeline import (
            extend_rolling_predictions,
            update_quality_score,
            update_signal_config,
        )

        new_pred = extend_rolling_predictions(new_test_end)

        # 3. 安全检查
        safety_msg = '首次训练'
        if old_pred is not None:
            is_safe, safety_msg = _check_prediction_safety(old_pred, new_pred)
            log.info(f"安全检查: {safety_msg}")

            if not is_safe:
                # 突变！回退
                _rollback_predictions(old_pred)
                alert = (
                    f"⚠️ 自动重训: 预测突变，已回退\n"
                    f"{safety_msg}\n"
                    f"请手动检查: python retrain_pipeline.py"
                )
                log.warning(alert)
                push_feishu(alert, dry_run)
                return True

        # 4. 安全，更新配置
        update_quality_score(new_pred)
        update_signal_config(new_test_end)

        # 5. 获取新 window 信息
        sg2 = SignalGenerator()
        freshness2 = sg2.check_model_freshness()
        window_num = freshness2.get('last_window', '?')

        notify = (
            f"🔄 自动重训完成\n"
            f"新增 Window {window_num}\n"
            f"{safety_msg}"
        )
        log.info(notify)
        push_feishu(notify, dry_run)
        return True

    except Exception as e:
        log.error(f"自动重训异常: {e}", exc_info=True)
        push_feishu(f"❌ 自动重训异常: {e}\n请手动检查", dry_run)
        return False


# ============ 信号生成 + 推送 ============

def generate_and_push(dry_run: bool = False, degraded: list = None):
    """生成信号 → 推送飞书"""
    from factor_lab.signal_generator import SignalGenerator
    from portfolio.live_portfolio import (
        load_live_holdings, save_live_holdings,
        get_current_prices, generate_live_instructions,
        check_stop_loss, get_daily_report, get_stock_names,
    )

    # 预检查: 预测缓存文件是否存在
    if not PRED_PKL.exists():
        # 尝试备用路径: 查找同目录下其他 pkl 文件
        alt_pkls = list(PRED_DIR.glob("*.pkl")) if PRED_DIR.exists() else []
        if alt_pkls:
            alt_names = [p.name for p in alt_pkls[:5]]
            msg = (f"❌ 预测缓存不存在: {PRED_PKL.name}\n"
                   f"可用文件: {', '.join(alt_names)}\n"
                   f"请检查 signal_config.yaml 配置或重新运行训练")
        else:
            msg = (f"❌ 预测缓存不存在: {PRED_PKL}\n"
                   f"请先运行: python retrain_pipeline.py")
        log.error(msg)
        push_feishu(msg, dry_run)
        return

    log.info("加载信号生成器...")
    sg = SignalGenerator()

    # 获取最新信号
    signal = sg.get_signal()
    if 'error' in signal:
        log.error(f"信号生成失败: {signal['error']}")
        push_feishu(f"❌ ML信号生成失败: {signal['error']}", dry_run)
        return

    log.info(f"信号日期: {signal['date']}, 状态: {signal['regime']}, TopK: {signal['effective_topk']}")

    # 信号时效性检查: 超过2个交易日标记为 stale
    signal_stale = False
    try:
        signal_date = datetime.strptime(signal['date'], '%Y-%m-%d').date()
        days_gap = (date.today() - signal_date).days
        if days_gap > 3:  # 日历日>3 约等于交易日>2
            signal_stale = True
            log.warning(f"信号已过期: 信号日期 {signal['date']}, 距今 {days_gap} 天")
    except (ValueError, KeyError):
        pass

    # 加载持仓
    holdings = load_live_holdings()

    # 更新调仓计数
    holdings['rebalance_day_count'] = holdings.get('rebalance_day_count', 0) + 1

    # 获取价格 (当前持仓 + 目标股票)
    all_instruments = list(set(
        list(holdings.get('positions', {}).keys()) + signal['target_stocks']
    ))
    prices, from_cache = get_current_prices(all_instruments)
    if not prices:
        log.error("获取价格失败")
        push_feishu("❌ 获取实时价格失败，请检查 BaoStock 连接", dry_run)
        return
    if from_cache:
        log.warning("使用缓存价格 (BaoStock不可用)")
        if degraded is None:
            degraded = []
        degraded.append("价格来自缓存(BaoStock不可用)")

    # 判断是否为调仓日 (每 5 个交易日)
    rebalance_every = 5
    is_rebalance = (holdings['rebalance_day_count'] % rebalance_every == 0)
    log.info(f"交易日计数: {holdings['rebalance_day_count']}, 调仓: {'是' if is_rebalance else '否'}")

    if is_rebalance:
        # ── 调仓日: 生成调仓指令 ──
        message = generate_live_instructions(signal, holdings, prices)

        # 保存待执行订单到 holdings
        target_set = set(signal['target_stocks'])
        current_set = set(holdings.get('positions', {}).keys())
        to_sell = list(current_set - target_set)
        to_buy = [c for c in signal['target_stocks'] if c not in current_set]

        holdings['pending_orders'] = {
            'sells': to_sell,
            'buys': {c: signal['scores'].get(c, 0) for c in to_buy},
        }
        holdings['last_signal_date'] = signal['date']
        save_live_holdings(holdings)

        log.info(f"调仓指令: 卖{len(to_sell)} 买{len(to_buy)}")
    else:
        # ── 非调仓日: 日报 ──
        message = get_daily_report(holdings, prices)

        # 止损检查
        alerts = check_stop_loss(holdings, prices)
        if alerts:
            alert_names = [f"{a['name']}({a['loss_pct']:.1%})" for a in alerts]
            message += f"\n\n🚨 止损触发: {', '.join(alert_names)}"
            message += "\n建议立即卖出上述股票"
            log.warning(f"止损预警: {alert_names}")

        save_live_holdings(holdings)

    # 信号过期提示
    if signal_stale:
        days_gap = (date.today() - signal_date).days
        message += f"\n\n⚠️ 信号已过期: 信号日期 {signal['date']}，距今 {days_gap} 天，请关注数据刷新是否正常"

    # 检查模型新鲜度
    freshness = sg.check_model_freshness()
    if freshness['is_stale']:
        message += f"\n\n⚠️ {freshness['message']}\n发送「重训」执行季度重训"

    # === Shadow Trading ===
    shadow_brief = _run_shadow_updates(signal, prices, dry_run)
    if shadow_brief:
        message += f"\n\n{shadow_brief}"

    # === Experiment Updates ===
    exp_brief = _run_experiment_updates(signal, prices, holdings, dry_run)
    if exp_brief:
        message += f"\n\n{exp_brief}"

    # === 降级摘要 (告知用户哪些 API 失败了) ===
    if degraded:
        message += "\n\n⚠️ 降级运行: " + " | ".join(degraded)

    push_feishu(message, dry_run)


def _run_shadow_updates(live_signal: dict, prices: dict,
                        dry_run: bool) -> str:
    """为所有 active shadow 生成信号并记录对比"""
    try:
        from shadow_manager import ShadowManager, calc_daily_returns
        from factor_lab.signal_generator import SignalGenerator

        sm = ShadowManager()
        active = sm.get_active_candidates()
        if not active:
            return ""

        log.info(f"Shadow 验证: {len(active)} 个活跃")
        briefs = []

        for sid, cand in active.items():
            try:
                config_path = cand.get('signal_config_path', '')
                sg_shadow = SignalGenerator(config_path)
                shadow_signal = sg_shadow.get_signal()

                if 'error' in shadow_signal:
                    log.warning(f"Shadow {sid} 信号生成失败: "
                                f"{shadow_signal['error']}")
                    continue

                sm.update_daily(sid, live_signal, shadow_signal, prices)

                perf = sm.get_performance(sid)
                # elapsed_days 已在 update_daily 中 +1
                elapsed = sm.registry.get(sid, {}).get('elapsed_days', 0)
                duration = cand.get('duration_days', 20)
                cum_ret = perf.get('cumulative_return', 0)
                overlap = perf.get('avg_overlap', 0)

                briefs.append(
                    f"  {sid}: 重叠{overlap:.0%} "
                    f"收益{cum_ret:+.1%} ({elapsed}/{duration}天)")
            except Exception as e:
                log.error(f"Shadow {sid} 更新失败: {e}")

        # 检查到期
        expired = sm.check_expired()
        for sid in expired:
            report = sm.get_expiry_report(sid)
            if report:
                push_feishu(f"📊 {report}", dry_run)
                log.info(f"Shadow {sid} 已到期，发送对比报告")

        if briefs:
            return "🧪 影子验证:\n" + "\n".join(briefs)
        return ""
    except Exception as e:
        log.error(f"Shadow 更新异常: {e}")
        return ""


def _run_experiment_updates(live_signal: dict, prices: dict,
                            holdings: dict, dry_run: bool) -> str:
    """为所有 active 实验盘生成信号并记录对比"""
    try:
        from experiment_manager import ExperimentManager
        from news_sentinel import NewsSentinel
        from portfolio.live_portfolio import get_stock_names

        em = ExperimentManager()
        active = em.get_active_experiments()
        if not active:
            return ""

        log.info(f"实验盘: {len(active)} 个活跃")

        # 1. 获取新闻数据 (所有实验共享一次)
        sentinel = NewsSentinel()

        # 宏观新闻
        macro_news = sentinel.fetch_macro_news()

        # 持仓列表 (用于宏观分析)
        stock_names = get_stock_names()
        holdings_list = []
        for code in holdings.get('positions', {}):
            holdings_list.append({
                "code": code,
                "name": stock_names.get(code, code),
            })

        # Claude 宏观分析
        macro_analysis = sentinel.analyze_macro(macro_news, holdings_list)
        log.info(f"宏观情绪: {macro_analysis.get('sentiment', 'N/A')}")

        # 候选股个股新闻
        candidate_codes = live_signal.get('target_stocks', [])[:30]
        stock_news = sentinel.fetch_stock_news(candidate_codes)

        # Claude 个股风险筛查
        risks = sentinel.screen_stock_risks(stock_news, stock_names)
        flagged = risks.get('flagged', [])
        if flagged:
            log.info(f"风险标记: {[f['code'] for f in flagged]}")

        # 2. 对每个实验: 应用过滤 + 记录
        briefs = []
        for eid, exp in active.items():
            try:
                exp_type = exp.get('experiment_type', '')

                if exp_type == 'sentiment':
                    # 情绪哨兵: 应用宏观+风险过滤
                    exp_signal = sentinel.apply_sentiment_filter(
                        live_signal, macro_analysis, risks)
                    extra = {
                        'macro_sentiment': exp_signal.get(
                            'macro_sentiment', ''),
                        'filtered_stocks': exp_signal.get(
                            'filtered_stocks', []),
                        'replacement_stocks': exp_signal.get(
                            'replacement_stocks', []),
                        'topk_adjustment': exp_signal.get(
                            'topk_adjustment', 0),
                    }
                else:
                    # 其他类型: 暂时直接用基线信号
                    exp_signal = live_signal
                    extra = {}

                em.update_daily(eid, live_signal, exp_signal, prices, extra)

                # 生成复盘文本
                review = em.get_daily_review(eid)
                if review:
                    briefs.append(review)
            except Exception as e:
                log.error(f"实验 {eid} 更新失败: {e}")

        # 3. 检查到期
        expired = em.check_expired()
        for eid in expired:
            report = em.get_expiry_report(eid)
            if report:
                push_feishu(f"📊 {report}", dry_run)
                log.info(f"实验 {eid} 已到期，发送报告")

        # 4. 组装推送文本
        if not briefs and not macro_analysis:
            return ""

        parts = []
        if briefs:
            parts.append(
                "━━━ 实验盘 (模拟,不影响实盘) ━━━\n"
                + "\n".join(briefs))

        macro_review = em.get_macro_review(
            list(active.keys())[0]) if active else ""
        if macro_review:
            parts.append(f"📰 {macro_review}")

        return "\n\n".join(parts)

    except ImportError as e:
        log.debug(f"实验盘模块未安装: {e}")
        return ""
    except Exception as e:
        log.error(f"实验盘更新异常: {e}", exc_info=True)
        return ""


# ============ 早间反思汇报 ============

def morning_reflection_report(dry_run: bool = False):
    """读取昨晚的反思结果，推送早间汇报"""
    from datetime import timedelta
    reflect_dir = PROJECT_DIR / "reflections"

    # 查找最近一次反思 (昨天或前天)
    reflection = None
    reflect_date = None
    for days_back in range(1, 4):
        d = (date.today() - timedelta(days=days_back)).isoformat()
        reflect_file = reflect_dir / f"{d}.json"
        if reflect_file.exists():
            try:
                with open(reflect_file, 'r', encoding='utf-8') as f:
                    reflection = json.load(f)
                reflect_date = d
                break
            except Exception:
                continue

    if not reflection:
        log.info("无最近反思记录，跳过晨报")
        return

    score = reflection.get('score', 0)
    summary = reflection.get('summary', '无')
    score_bar = '★' * score + '☆' * (10 - score)

    parts = [
        f"🌅 每日反思汇报 ({reflect_date})",
        f"评分: {score_bar} ({score}/10)",
        f"总结: {summary}",
    ]

    # 用户体验
    ux = reflection.get('user_experience', {})
    if ux:
        highlights = ux.get('highlights', [])
        issues = ux.get('issues', [])
        if highlights:
            parts.append(f"亮点: {', '.join(highlights[:3])}")
        if issues:
            parts.append(f"问题: {', '.join(issues[:3])}")

    # 稳定性
    stab = reflection.get('stability', {})
    if stab:
        errs = stab.get('errors_count', 0)
        restarts = stab.get('restarts', 0)
        if errs or restarts:
            parts.append(f"稳定性: {errs} 错误, {restarts} 重启")

    # 代码改进
    improvements = reflection.get('code_improvements', [])
    if improvements:
        parts.append(f"代码改进建议: {len(improvements)} 项")
        for imp in improvements[:3]:
            parts.append(f"  - [{imp.get('priority', '?')}] {imp.get('description', '')}")

    # 自动修复
    auto_fixes = reflection.get('auto_fixes', [])
    if auto_fixes and any(not f.get('error') for f in auto_fixes):
        parts.append(f"自动修复: 已执行 {len(auto_fixes)} 项")

    # Action items
    actions = reflection.get('action_items', [])
    if actions:
        parts.append("今日待办:")
        for item in actions[:5]:
            parts.append(f"  · {item}")

    message = "\n".join(parts)
    push_feishu(message, dry_run)
    log.info(f"反思晨报已推送: {reflect_date}, 评分 {score}/10")


# ============ 飞书推送 ============

def push_feishu(message: str, dry_run: bool = False):
    """推送消息到飞书"""
    if dry_run:
        log.info(f"[DRY RUN] 飞书消息:\n{message}")
        return

    try:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        app_id = os.environ.get("FEISHU_APP_ID_1", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET_1", "")
        user_id = os.environ.get("FEISHU_USER_OPEN_ID", "")

        if not all([app_id, app_secret, user_id]):
            log.error("飞书凭证未配置，消息未发送")
            log.info(f"消息内容:\n{message}")
            return

        client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .build()

        req = CreateMessageRequest.builder() \
            .receive_id_type("open_id") \
            .request_body(CreateMessageRequestBody.builder()
                .receive_id(user_id)
                .msg_type("text")
                .content(json.dumps({"text": message}))
                .build()) \
            .build()

        resp = client.im.v1.message.create(req)
        if resp.success():
            log.info("飞书消息发送成功")
        else:
            log.error(f"飞书发送失败: {resp.code} - {resp.msg}")
    except Exception as e:
        log.error(f"飞书推送异常: {e}")
        log.info(f"消息内容:\n{message}")


# ============ 主入口 ============

def _handle_shadow_commands(args) -> bool:
    """处理 shadow 相关 CLI 命令，返回 True 表示已处理"""
    from shadow_manager import ShadowManager

    if args.shadow_status:
        sm = ShadowManager()
        print(sm.get_status_text())
        return True

    if args.promote_shadow:
        sm = ShadowManager()
        sm.promote(args.promote_shadow)
        print(f"请运行 python retrain_pipeline.py 完成切换")
        return True

    if args.reject_shadow:
        sm = ShadowManager()
        sm.reject(args.reject_shadow)
        return True

    if args.extend_shadow:
        sid, extra = args.extend_shadow
        sm = ShadowManager()
        sm.extend(sid, int(extra))
        return True

    if args.archive_shadow:
        sm = ShadowManager()
        sm.archive(args.archive_shadow)
        print(f"旧模型已封存，自动重训恢复正常")
        return True

    if args.rollback_shadow:
        sm = ShadowManager()
        sm.rollback(args.rollback_shadow)
        print(f"已回退到旧基线，signal_config + predictions 已恢复")
        return True

    return False


def _handle_experiment_commands(args) -> bool:
    """处理 experiment 相关 CLI 命令，返回 True 表示已处理"""
    from experiment_manager import ExperimentManager

    if args.experiment_status:
        em = ExperimentManager()
        print(em.get_status_text())
        return True

    if args.create_sentiment_experiment:
        em = ExperimentManager()
        eid = em.create_experiment(
            name="情绪哨兵",
            experiment_type="sentiment",
            config={
                "macro_topk_adjust": True,
                "risk_filter": True,
                "bearish_topk_delta": -4,
                "bullish_topk_delta": 2,
            },
            reason="宏观情绪 + 个股风险过滤实验",
            duration_days=30,
        )
        print(f"已创建实验 {eid}")
        return True

    if args.reject_experiment:
        em = ExperimentManager()
        em.reject(args.reject_experiment)
        return True

    if args.extend_experiment:
        eid, extra = args.extend_experiment
        em = ExperimentManager()
        em.extend(eid, int(extra))
        return True

    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='每日信号推送')
    parser.add_argument('--dry-run', action='store_true', help='只生成不推送')
    parser.add_argument('--force', action='store_true', help='忽略交易日检查')
    parser.add_argument('--morning-report', action='store_true',
                        help='推送早间反思汇报')
    parser.add_argument('--shadow-status', action='store_true',
                        help='查看影子验证状态')
    parser.add_argument('--promote-shadow', type=str, metavar='ID',
                        help='晋升 shadow (如 shadow_001)')
    parser.add_argument('--reject-shadow', type=str, metavar='ID',
                        help='拒绝 shadow (如 shadow_001)')
    parser.add_argument('--extend-shadow', nargs=2, metavar=('ID', 'DAYS'),
                        help='延长验证 (如 shadow_001 10)')
    parser.add_argument('--archive-shadow', type=str, metavar='ID',
                        help='封存反转 shadow (确认新模型更优)')
    parser.add_argument('--rollback-shadow', type=str, metavar='ID',
                        help='回退到旧基线 (反转实验证明旧模型更优)')
    # Experiment 管理命令
    parser.add_argument('--experiment-status', action='store_true',
                        help='查看实验盘状态')
    parser.add_argument('--create-sentiment-experiment', action='store_true',
                        help='创建情绪哨兵实验')
    parser.add_argument('--reject-experiment', type=str, metavar='ID',
                        help='终止实验 (如 exp_001)')
    parser.add_argument('--extend-experiment', nargs=2, metavar=('ID', 'DAYS'),
                        help='延长实验 (如 exp_001 15)')
    args = parser.parse_args()

    # Shadow 管理命令
    if any([args.shadow_status, args.promote_shadow,
            args.reject_shadow, args.extend_shadow,
            args.archive_shadow, args.rollback_shadow]):
        _handle_shadow_commands(args)
        return

    # Experiment 管理命令
    if any([args.experiment_status, args.create_sentiment_experiment,
            args.reject_experiment, args.extend_experiment]):
        _handle_experiment_commands(args)
        return

    # 早间反思汇报
    if args.morning_report:
        log.info("推送早间反思汇报...")
        morning_reflection_report(dry_run=args.dry_run)
        return

    log.info("=" * 50)
    log.info("Daily Runner 启动")
    log.info("=" * 50)

    degraded = []  # 收集降级事件

    try:
        # 交易日检查
        if not args.force and not is_trading_day():
            log.info("今天非交易日，跳过")
            return

        # 刷新数据 (失败则用旧数据继续，但记录警告)
        if not refresh_daily_data():
            log.warning("数据刷新失败，将使用缓存数据继续")
            degraded.append("BaoStock数据刷新失败(用缓存)")

        # 增量预测: 利用已有模型扩展预测到最新数据日期
        if not _update_daily_predictions() and degraded:
            # 数据刷新失败且增量预测也没更新才告警
            degraded.append("增量预测未更新")

        # 自动重训检查 (距 pred_end > 60天则触发)
        _check_auto_retrain(dry_run=args.dry_run)

        # 生成信号 + 推送 (传入降级信息)
        generate_and_push(dry_run=args.dry_run, degraded=degraded)

        log.info("Daily Runner 完成")
    except Exception as e:
        log.error(f"Daily Runner 异常退出: {e}", exc_info=True)
        try:
            import traceback
            tb = traceback.format_exc()[-300:]
            push_feishu(
                f"❌ Daily Runner 异常退出:\n{e}\n\n"
                f"请检查日志: logs/daily_runner.log\n{tb}",
                dry_run=args.dry_run
            )
        except Exception:
            pass


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()
