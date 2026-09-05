#!/usr/bin/env python3
"""自动因子挖掘 — Claude Agent 驱动，每日渐进式运行

用法:
    python -m factor_lab.factor_miner                        # 每日渐进式挖掘 (默认)
    python -m factor_lab.factor_miner --daily --smoke-test   # 快速验证 (2方向, 1h)
    python -m factor_lab.factor_miner --evolved              # 单次全量演化式挖掘
    python -m factor_lab.factor_miner --evolved --smoke-test # 快速验证 (3方向×2轮)
    python -m factor_lab.factor_miner --legacy               # 旧流程
    python -m factor_lab.factor_miner --dry-run              # 不推送不回测
    python -m factor_lab.factor_miner --report               # 查看历史
    python -m factor_lab.factor_miner --accept run_001       # 人工接受因子
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from factor_lab.utils import json_default as _json_default
from feishu_target import resolve_open_id as _resolve_open_id

MINING_DIR = PROJECT_DIR / "factor_lab" / "mining_results"
RUNS_DIR = MINING_DIR / "runs"
DISCOVERIES_DIR = MINING_DIR / "discoveries"
LOCK_FILE = MINING_DIR / ".lock"

MAX_CYCLES = 4  # 最多迭代轮数 (0-2: 种子轮转, 3: crossover)
BATCH_SIZE = 10  # 每轮生成因子数
TOTAL_TIMEOUT = 3600  # 总运行超时 (秒)


def _ensure_dirs():
    for d in [MINING_DIR, RUNS_DIR, DISCOVERIES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _pid_alive(pid: int) -> bool:
    """跨平台检查进程是否存活（Windows 上 os.kill(pid, 0) 会误杀进程）"""
    if os.name == 'nt':
        import subprocess
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return str(pid) in (result.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock() -> bool:
    """lockfile 防并发"""
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            if _pid_alive(pid):
                return False
        except (ValueError, PermissionError):
            pass  # 旧锁，PID 无效
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock():
    if LOCK_FILE.exists():
        LOCK_FILE.unlink(missing_ok=True)


def _next_run_id() -> str:
    """生成下一个 run ID"""
    existing = sorted(RUNS_DIR.glob("run_*.json"))
    if existing:
        last_num = int(existing[-1].stem.split("_")[1])
        return f"run_{last_num + 1:03d}"
    return "run_001"


def _init_qlib():
    """初始化 Qlib (只执行一次)"""
    import yaml
    import qlib
    from qlib.constant import REG_CN
    cfg_path = PROJECT_DIR / "config" / "signal_config.yaml"
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    universe = cfg.get('instruments', 'csi300')
    data_dir = f"~/.qlib/qlib_data/cn_data_{'bs' if universe == 'csi300' else universe}"
    try:
        qlib.init(provider_uri=data_dir, region=REG_CN)
    except Exception:
        pass  # 已初始化


def _get_all_preset_names() -> list[str]:
    """获取所有预设中的因子名"""
    from factor_lab.factors.presets import FACTOR_PRESETS
    names = set()
    for preset in FACTOR_PRESETS.values():
        extra = preset["extra_factors"]
        if callable(extra):
            try:
                extra = extra()
            except Exception:
                continue
        if isinstance(extra, list):
            for n, _ in extra:
                names.add(n)
    return sorted(names)


def _push_feishu(message: str):
    """飞书推送 (复用 daily_runner 的凭证)"""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_DIR / ".env")

        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        app_id = os.environ.get("FEISHU_APP_ID_1", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET_1", "")
        user_id = _resolve_open_id()

        if not all([app_id, app_secret, user_id]):
            print(f"  [feishu] 凭证未配置，消息:\n{message}")
            return

        client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
        req = CreateMessageRequest.builder() \
            .receive_id_type("open_id") \
            .request_body(CreateMessageRequestBody.builder()
                .receive_id(user_id)
                .msg_type("text")
                .content(json.dumps({"text": message}))
                .build()) \
            .build()
        client.im.v1.message.create(req)
        print("  [feishu] 消息已发送")
    except Exception as e:
        print(f"  [feishu] 推送失败: {e}")
        print(f"  消息内容:\n{message}")


def _save_discoveries(promising_factors: list[dict], run_id: str):
    """保存非冗余 promising 因子到 discoveries 目录"""
    for pf in promising_factors:
        if not pf.get("is_redundant"):
            disc_file = DISCOVERIES_DIR / f"{run_id}_{pf['name']}.json"
            disc_file.write_text(
                json.dumps(pf, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )


def _check_beat_baseline(backtest) -> bool:
    """检查回测是否超越 baseline"""
    return (backtest and isinstance(backtest, dict)
            and backtest.get("improvement", {}).get("is_better", False))


def _check_overfitting(backtest: dict, run_result: dict) -> dict:
    """多重检验校正 — DSR (紧缩夏普比率)

    2026-09-03 新增。原验收标准只有 "holdout 上 beat_baseline"，
    对"试了多少次才挑出这个"毫无校正。一轮 --evolved 是 8 方向 × 6 轮 × 2 候选，
    几百次试验起步 —— 从一堆真实夏普为 0 的随机策略里挑最大值，
    那个最大值必然显著大于 0。实测: 200 个纯随机策略里最好的一个，
    年化夏普能到 2.0。

    run_006/run_010 的 22 个因子(样本内 +7.4% / 样本外 -12.9%)正是这个机制。

    Returns:
        stats_overfit.evaluate 的结果; 无法检验时返回 {} (不阻断，但会记录原因)
    """
    from factor_lab.stats_overfit import deflated_sharpe, format_report

    rets = (backtest or {}).get("candidate_returns") or []
    if len(rets) < 8:
        return {"note": "无日收益序列，跳过 DSR 检验"}

    # 试验次数: 本轮实际测试过的因子数 (含被拒的) —— 每个都是一次"试验"
    n_trials = max(int(run_result.get("factors_tested", 0)), 1)

    # 各次试验夏普的标准差: 手上只有本轮汇总，用候选与基线的差异做保守估计。
    # 低估离散度会低估运气门槛，故设下限 0.3，避免把门槛调得过松。
    bl_sr = (backtest.get("baseline") or {}).get("sharpe", 0.0)
    cd_sr = (backtest.get("candidate") or {}).get("sharpe", 0.0)
    sr_std = max(0.3, abs(cd_sr - bl_sr) * 2)

    result = {"dsr": deflated_sharpe(rets, n_trials=n_trials, sr_std_ann=sr_std)}
    result["passed"] = bool(result["dsr"].get("passed"))
    result["report"] = format_report(result)
    return result


def _build_discovery_message(run_result: dict, backtest: dict) -> str:
    """构建飞书通知消息"""
    run_id = run_result.get("run_id", "?")
    dt = run_result.get("date", "?")
    tested = run_result.get("factors_tested", 0)
    promising = run_result.get("promising_count", 0)

    lines = [
        f"🔬 因子挖掘 — 发现候选因子",
        "",
        f"运行: #{run_id} ({dt})",
        f"测试: {tested} 因子 → {promising} 个有效",
    ]

    # 最佳发现
    best_factors = run_result.get("promising_factors", [])
    if best_factors:
        lines.append("")
        lines.append("【最佳发现】")
        for f in best_factors[:3]:
            lines.append(f"  {f['name']} (ICIR={f.get('icir', 0):.3f})")
            lines.append(f"  {f.get('expr', '')[:80]}")
            lines.append(f"  假说: {f.get('hypothesis', '')[:60]}")
            lines.append("")

    # 回测对比
    if backtest and "baseline" in backtest:
        bl = backtest["baseline"]
        cd = backtest["candidate"]
        imp = backtest["improvement"]
        lines.append("回测对比 (vs M01-LGB-D3v3r-v2602):")
        lines.append(f"  M01:  Sharpe {bl['sharpe']:.3f} | MDD {bl['mdd']:.2%} | Return {bl['return']:.0%}")
        lines.append(f"  候选: Sharpe {cd['sharpe']:.3f} | MDD {cd['mdd']:.2%} | Return {cd['return']:.0%}")
        mark = "✅" if imp["is_better"] else "❌"
        lines.append(f"  提升: {imp['sharpe_delta']:+.3f} Sharpe {mark}")

    lines.append("")
    lines.append("请人工审核后决定是否纳入。")
    lines.append(f"接受命令: python -m factor_lab.factor_miner --accept {run_id}")

    return "\n".join(lines)


def run_mining(dry_run: bool = False, skip_backtest: bool = False):
    """执行一次完整的挖掘周期"""
    from factor_lab.mining.context import MiningContext
    from factor_lab.mining.hypothesis import (
        generate_hypotheses, mutate_factors, crossover_hypotheses,
        SEED_CATEGORIES,
    )
    from factor_lab.mining.validator import validate_expression, validate_with_qlib
    from factor_lab.mining.evaluator import evaluate_candidates, check_redundancy

    t_start = time.time()
    run_id = _next_run_id()
    today = date.today().isoformat()

    print(f"\n{'='*60}")
    print(f"  因子挖掘 — {run_id} ({today})")
    print(f"{'='*60}")

    # 1. 加载 context
    ctx = MiningContext()
    focus = ctx.get_focus_areas()
    existing = ctx.get_all_tried_names() + _get_all_preset_names()
    print(f"  已知因子: {len(existing)} 个")
    print(f"  探索方向: {focus[:80]}...")

    # 初始化 Qlib
    _init_qlib()

    all_hypotheses = []
    all_valid = []
    all_eval_results = []
    promising_factors = []
    non_redundant_factors = []
    failed_approaches = []
    near_miss_factors = []  # 差一点的因子 (供 mutation)

    # 获取历史 discoveries (供 crossover)
    discoveries = ctx.get_discoveries()
    seed_index = 0  # 独立种子计数器 (不被 mutation/crossover 打断)

    # 2. 迭代生成和评估
    for cycle in range(MAX_CYCLES):
        if time.time() - t_start > TOTAL_TIMEOUT:
            print(f"  [超时] 总运行超过 {TOTAL_TIMEOUT}s")
            break

        print(f"\n  --- 第 {cycle + 1}/{MAX_CYCLES} 轮 ---")

        # 决定本轮策略: 种子轮转 / mutation / crossover
        # 取出本轮 mutation 目标 (独立副本)，清空 near_miss 供本轮重新收集
        mutation_targets = near_miss_factors.copy() if (near_miss_factors and cycle > 0) else []
        near_miss_factors = []

        if mutation_targets:
            # 有 near-miss → mutation 优先
            print(f"  [mutation] 对 {len(mutation_targets)} 个 near-miss 因子靶向修复")
            hypotheses = mutate_factors(mutation_targets, ctx.load())
        elif cycle == MAX_CYCLES - 1 and len(discoveries) >= 3:
            # 最后一轮 + 有足够历史 → crossover
            print(f"  [crossover] 基于 {len(discoveries)} 个历史因子重组")
            hypotheses = crossover_hypotheses(discoveries, ctx.load(), BATCH_SIZE)
        else:
            # 种子轮转生成
            seed = SEED_CATEGORIES[seed_index % len(SEED_CATEGORIES)]
            seed_index += 1
            print(f"  [seed] 方向: {seed['focus']}")
            hypotheses = generate_hypotheses(
                ctx.load(), existing, batch_size=BATCH_SIZE,
                seed=seed,
            )

        print(f"  生成假说: {len(hypotheses)} 个")
        all_hypotheses.extend(hypotheses)

        if not hypotheses:
            failed_approaches.append(f"cycle_{cycle+1}: Claude 未返回有效假说")
            continue

        # 静态验证
        static_valid = []
        for h in hypotheses:
            ok, err = validate_expression(h["name"], h["expr"])
            if ok:
                static_valid.append(h)
            else:
                print(f"    ✗ {h['name']}: {err}")
        print(f"  静态验证通过: {len(static_valid)}/{len(hypotheses)}")

        # 动态验证
        valid = []
        for h in static_valid:
            ok, err = validate_with_qlib(h["name"], h["expr"])
            if ok:
                valid.append(h)
                existing.append(h["name"])
            else:
                print(f"    ✗ {h['name']}: {err}")
        print(f"  动态验证通过: {len(valid)}/{len(static_valid)}")
        all_valid.extend(valid)

        if not valid:
            failed_approaches.append(f"cycle_{cycle+1}: 所有因子验证失败")
            continue

        # IC/ICIR 评估
        factor_tuples = [(h["name"], h["expr"]) for h in valid]
        eval_df = evaluate_candidates(factor_tuples)
        all_eval_results.append(eval_df)
        print(f"  评估完成: {len(eval_df)} 个因子")

        if not eval_df.empty:
            for _, row in eval_df.iterrows():
                tag = "★" if row.get("is_excellent") else ("●" if row.get("is_promising") else "○")
                print(f"    {tag} {row['factor']}: ICIR={row['ICIR']:.3f}")

        # 筛选 promising
        promising_df = eval_df[eval_df["is_promising"]] if not eval_df.empty else eval_df
        if promising_df.empty:
            failed_approaches.append(f"cycle_{cycle+1}: 无 promising 因子 (|ICIR|<0.5)")
            # 收集 near-miss (0.3 ≤ |ICIR| < 0.5) 供下轮 mutation
            if not eval_df.empty:
                nm_df = eval_df[(eval_df["ICIR"].abs() >= 0.3) & (eval_df["ICIR"].abs() < 0.5)]
                for _, row in nm_df.iterrows():
                    h_info = next((h for h in valid if h["name"] == row["factor"]), {})
                    near_miss_factors.append({
                        "name": row["factor"],
                        "expr": h_info.get("expr", ""),
                        "hypothesis": h_info.get("hypothesis", ""),
                        "icir": float(row["ICIR"]),
                        "is_redundant": False,
                        "max_corr": 0,
                    })
                if near_miss_factors:
                    print(f"  发现 {len(near_miss_factors)} 个 near-miss 因子 (0.3≤|ICIR|<0.5)")
            continue

        # 冗余检测
        promising_tuples = [(row["factor"], next(h["expr"] for h in valid if h["name"] == row["factor"]))
                            for _, row in promising_df.iterrows()]
        redundancy = check_redundancy(promising_tuples)
        print(f"  冗余检测:")
        for name, info in redundancy.items():
            tag = "冗余" if info["is_redundant"] else "独立"
            print(f"    {tag}: {name} (max_corr={info['max_corr']:.3f} vs {info['most_correlated']})")

        non_redundant = [(n, e) for n, e in promising_tuples if not redundancy[n]["is_redundant"]]

        # 收集 promising 因子详情
        for n, e in promising_tuples:
            h_info = next((h for h in valid if h["name"] == n), {})
            icir = float(eval_df[eval_df["factor"] == n]["ICIR"].iloc[0]) if not eval_df[eval_df["factor"] == n].empty else 0
            factor_detail = {
                "name": n,
                "expr": e,
                "icir": icir,
                "hypothesis": h_info.get("hypothesis", ""),
                "category": h_info.get("category", ""),
                "is_redundant": redundancy.get(n, {}).get("is_redundant", False),
                "max_corr": redundancy.get(n, {}).get("max_corr", 0),
                "most_correlated": redundancy.get(n, {}).get("most_correlated", ""),
            }
            promising_factors.append(factor_detail)
            # 冗余且 |ICIR| ≥ 0.3 的 promising 因子也收集为 near-miss (供 mutation 差异化)
            if factor_detail["is_redundant"] and abs(factor_detail["icir"]) >= 0.3:
                near_miss_factors.append(factor_detail)

        non_redundant_factors.extend(non_redundant)
        if non_redundant:
            print(f"  非冗余 promising 因子: {len(non_redundant)} 个")
            break  # 有发现，进入回测

        failed_approaches.append(f"cycle_{cycle+1}: promising 因子全部冗余")
        if near_miss_factors:
            print(f"  收集 {len(near_miss_factors)} 个 near-miss/冗余因子待 mutation")

    # 3. 回测对比
    backtest = None
    if non_redundant_factors and not skip_backtest and not dry_run:
        print(f"\n  === 全量 Rolling 回测 ({len(non_redundant_factors)} 新因子) ===")
        try:
            from factor_lab.mining.backtest_runner import run_comparison
            backtest = run_comparison(non_redundant_factors)
        except Exception as e:
            print(f"  [backtest] 回测失败: {e}")
            backtest = {"error": str(e)}

    # 4. 保存审计记录
    run_result = {
        "run_id": run_id,
        "date": today,
        "factors_tested": len(all_valid),
        "hypotheses_generated": len(all_hypotheses),
        "hypotheses": [
            {"name": h["name"], "expr": h["expr"],
             "hypothesis": h["hypothesis"], "category": h.get("category", "")}
            for h in all_hypotheses
        ],
        "promising_count": len(promising_factors),
        "promising_factors": promising_factors,
        "non_redundant": [{"name": n, "expr": e} for n, e in non_redundant_factors],
        "backtest": backtest,
        "failed_approaches": failed_approaches,
        "focus": focus,
        "total_time": round(time.time() - t_start, 1),
    }

    # 生成下次方向建议
    next_directions = _suggest_next_directions(run_result)
    run_result["next_directions"] = next_directions

    run_file = RUNS_DIR / f"{run_id}.json"
    run_file.write_text(json.dumps(run_result, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"\n  审计记录: {run_file}")

    _save_discoveries(promising_factors, run_id)

    # 5. 更新 context
    ctx.update_after_run(run_result)
    print(f"  Agent 记忆已更新")

    # 6. 通知
    beat_baseline = _check_beat_baseline(backtest)
    run_result["beat_baseline"] = beat_baseline

    # 多重检验校正: beat_baseline 只说明"比基线好"，DSR 回答"扣掉试了几百次
    # 带来的运气成分后还剩多少"。两道都过才写入 mined.py。
    overfit = _check_overfitting(backtest, run_result) if beat_baseline else {}
    run_result["overfit_check"] = overfit
    if beat_baseline and overfit and not overfit.get("passed", True):
        print("\n" + overfit.get("report", ""))
        print("  ⚠ 未通过多重检验校正 —— 不写入 mined.py，仅记录供人工复核")
        beat_baseline = False
        run_result["beat_baseline"] = False
        run_result["rejected_by"] = "DSR"

    if beat_baseline and not dry_run:
        # 写入 mined.py + 自动创建 shadow 验证
        _accept_factors_to_mined(non_redundant_factors, run_id)
        shadow_id = _create_shadow_for_run(run_id, backtest)
        msg = _build_discovery_message(run_result, backtest)
        if overfit.get("report"):
            msg += "\n\n" + overfit["report"]
        if shadow_id:
            msg += f"\n\n已自动创建影子验证: {shadow_id} (20交易日)"
        _push_feishu(msg)
        from notifications import write_notification
        write_notification("mining", f"因子晋升: #{run_id}", msg)
    elif promising_factors and not dry_run:
        # 有 promising 但未 beat baseline，也发一条简要通知
        names = [f["name"] for f in promising_factors if not f.get("is_redundant")]
        if names:
            msg = (f"🔬 因子挖掘 #{run_id}\n"
                   f"发现 {len(names)} 个非冗余因子: {', '.join(names[:5])}\n"
                   f"{'回测未超越 M01' if backtest else '未执行回测'}")
            _push_feishu(msg)
            from notifications import write_notification
            write_notification("mining", f"发现非冗余因子: #{run_id}", msg)

    # 打印总结
    print(f"\n{'='*60}")
    print(f"  总结: 测试 {len(all_valid)} 因子, "
          f"promising {len(promising_factors)}, "
          f"非冗余 {len(non_redundant_factors)}")
    if backtest and "baseline" in backtest:
        print(f"  回测: Sharpe {backtest['candidate']['sharpe']:.3f} "
              f"(baseline {backtest['baseline']['sharpe']:.3f}, "
              f"delta {backtest['improvement']['sharpe_delta']:+.3f})")
    print(f"  耗时: {time.time() - t_start:.1f}s")
    print(f"{'='*60}")


def _suggest_next_directions(run_result: dict) -> list[str]:
    """根据本次结果建议下次方向"""
    directions = []

    # 检查哪些类别有 promising
    categories_tried = set()
    categories_promising = set()
    for h in run_result.get("hypotheses", []):
        cat = h.get("category", "unknown")
        categories_tried.add(cat)
    for p in run_result.get("promising_factors", []):
        cat = p.get("category", "unknown")
        categories_promising.add(cat)

    all_categories = {"volatility", "momentum", "liquidity", "value", "reversal", "microstructure"}
    untried = all_categories - categories_tried
    if untried:
        directions.append(f"未探索类别: {', '.join(untried)}")

    if categories_promising:
        directions.append(f"有潜力的类别: {', '.join(categories_promising)} (可深入变体)")

    # 默认方向
    default_dirs = [
        "日内波动结构 (开盘/收盘/最高/最低的相对位置)",
        "成交量形态 (量价时序关系的高阶特征)",
        "估值动量交叉因子 (PE/PB 的变化率)",
        "市值变化率因子 (市值的短期动量)",
        "换手率异常检测 (当前换手 vs 历史分位)",
    ]
    for d in default_dirs:
        if len(directions) < 8:
            directions.append(d)

    return directions


def show_report():
    """展示挖掘历史报告"""
    from factor_lab.mining.context import MiningContext, INDEX_FILE

    ctx = MiningContext()
    print(ctx.load())

    if INDEX_FILE.exists():
        index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        print(f"\n总运行: {len(index)} 次")
        for rid, info in sorted(index.items()):
            beat = "✅" if info.get("beat_baseline") else "  "
            print(f"  {beat} {rid} ({info.get('date', '?')}): "
                  f"tested={info.get('factors_tested', 0)}, "
                  f"promising={info.get('promising_count', 0)}")


def _accept_factors_to_mined(non_redundant: list, run_id: str):
    """将非冗余因子写入 factors/mined.py"""
    mined_file = PROJECT_DIR / "factor_lab" / "factors" / "mined.py"

    if mined_file.exists():
        content = mined_file.read_text(encoding="utf-8")
    else:
        content = ('"""被接受的挖掘因子"""\n\n'
                   'MINED_FACTORS = []\n\n\n'
                   'def get_all_exprs():\n    return MINED_FACTORS\n')

    existing_names = set()
    import ast
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "MINED_FACTORS":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Tuple) and len(elt.elts) >= 1:
                                    if isinstance(elt.elts[0], ast.Constant):
                                        existing_names.add(elt.elts[0].value)
    except Exception:
        pass

    new_entries = []
    for f in non_redundant:
        name = f["name"] if isinstance(f, dict) else f[0]
        expr = f["expr"] if isinstance(f, dict) else f[1]
        if name not in existing_names:
            new_entries.append((name, expr))

    if not new_entries:
        print("  所有因子已存在于 mined.py")
        return

    lines = [f'    ({repr(n)}, {repr(e)}),  # accepted from {run_id}'
             for n, e in new_entries]
    insert_text = "\n".join(lines)

    if "MINED_FACTORS = []" in content:
        content = content.replace(
            "MINED_FACTORS = []",
            f"MINED_FACTORS = [\n{insert_text}\n]",
        )
    elif "MINED_FACTORS = [" in content:
        # 精确找到 MINED_FACTORS 列表的结尾 `]`，而不用 rfind (会匹配文件最后一个])
        start = content.index("MINED_FACTORS = [")
        # 从 MINED_FACTORS 开始位置向后找第一个独立的 ]\n
        search_from = start + len("MINED_FACTORS = [")
        idx = content.index("\n]", search_from)
        content = content[:idx] + "\n" + insert_text + content[idx:]

    mined_file.write_text(content, encoding="utf-8")
    print(f"  已写入 {len(new_entries)} 个因子到 mined.py")


def _create_shadow_for_run(run_id: str, backtest: dict) -> str:
    """为 beat_baseline 的 run 创建 shadow candidate + retrain"""
    try:
        sys.path.insert(0, str(PROJECT_DIR))
        from shadow_manager import ShadowManager

        sm = ShadowManager()
        imp = backtest.get("improvement", {}) if isinstance(backtest, dict) else {}
        reason = f"Sharpe {imp.get('sharpe_delta', 0):+.3f}"
        shadow_id = sm.create_candidate(
            source=f"factor_miner:{run_id}",
            config_overrides={"preset": "alpha158_val_mined"},
            reason=reason,
            duration_days=20,
        )
        # 为 shadow 生成独立预测
        print(f"  [shadow] 正在 retrain {shadow_id}...")
        sm.retrain_for_shadow(shadow_id)
        print(f"  已创建影子验证: {shadow_id}")
        return shadow_id
    except Exception as e:
        print(f"  [shadow] 创建失败: {e}")
        return ""


def accept_run(run_id: str):
    """接受某次运行的非冗余因子 → 写入 mined.py + 创建 shadow 验证"""
    run_file = RUNS_DIR / f"{run_id}.json"
    if not run_file.exists():
        print(f"运行记录不存在: {run_file}")
        return

    run = json.loads(run_file.read_text(encoding="utf-8"))
    non_redundant = run.get("non_redundant", [])
    if not non_redundant:
        print(f"{run_id} 没有非冗余因子可接受")
        return

    # 写入 mined.py
    _accept_factors_to_mined(non_redundant, run_id)
    print(f"已接受 {len(non_redundant)} 个因子:")
    for f in non_redundant:
        name = f["name"] if isinstance(f, dict) else f[0]
        expr = f["expr"] if isinstance(f, dict) else f[1]
        print(f"  + {name}: {expr[:60]}")

    # 创建 shadow 验证
    backtest = run.get("backtest", {})
    shadow_id = _create_shadow_for_run(run_id, backtest)
    if shadow_id:
        print(f"\n已创建影子验证 {shadow_id} (20交易日)")
        print("验证到期后运行:")
        print(f"  python daily_runner.py --promote-shadow {shadow_id}")
    else:
        print("\n注意: shadow 创建失败，因子已写入 mined.py 但未启动验证")


# ============================================================================
# 演化式挖掘 (QuantaAlpha 风格, --evolved 默认)
# ============================================================================

EVOLVED_TOTAL_TIMEOUT = 28800  # 演化式挖掘总超时 (8h, 8方向×6轮需~5h)


def run_evolved_mining(dry_run: bool = False, smoke_test: bool = False):
    """演化式因子挖掘 (QuantaAlpha 风格)

    流程:
    1. Planning: Claude 生成 N 个搜索方向
    2. Round 0 (Original): 每个方向 → 5步循环 → TrajectoryPool
    3. Round 1+ (Mutation/Crossover 交替): 演化搜索
    4. Final: FactorPool 量化准入因子 → D_expand 最终验证 → shadow/mined.py
    """
    from factor_lab.quanta.config import EVOLVED_N_DIRECTIONS, EVOLVED_N_ROUNDS
    from factor_lab.quanta.trajectory import TrajectoryPool, Trajectory
    from factor_lab.quanta.factor_pool import FactorPool
    from factor_lab.quanta.evolution import (
        mutate_trajectory, get_mutation_targets,
        get_crossover_candidates, select_crossover_groups, crossover_trajectories,
    )
    from factor_lab.quanta import idea_agent, factor_agent, eval_agent
    from factor_lab.mining.planning_agent import plan_directions
    from factor_lab.mining.context import MiningContext

    t_start = time.time()
    run_id = _next_run_id()
    today = date.today().isoformat()

    n_dirs = 3 if smoke_test else EVOLVED_N_DIRECTIONS
    n_rounds = 2 if smoke_test else EVOLVED_N_ROUNDS

    print(f"\n{'='*60}")
    print(f"  演化式因子挖掘 — {run_id} ({today})")
    print(f"  模式: {'smoke-test' if smoke_test else 'evolved'}")
    print(f"  方向: {n_dirs} | 轮次: {n_rounds}")
    print(f"{'='*60}")

    # 初始化 Qlib
    _init_qlib()

    # 1. 初始化 TrajectoryPool + FactorPool (支持断点恢复)
    pool_dir = MINING_DIR / "trajectory_pool" / run_id
    pool = TrajectoryPool(save_dir=pool_dir)
    pool.load()  # 尝试从上次断点恢复

    factor_pool = FactorPool(save_dir=pool_dir)
    factor_pool.load()

    if pool.size > 0:
        print(f"  从断点恢复: {pool.size} 条轨迹, {factor_pool.size} 个入池因子")

    # 2. Planning: 生成搜索方向 (优先从断点恢复)
    print(f"\n  === Phase 1: 方向规划 ===")
    ctx = MiningContext()
    context_text = ctx.load()
    directions_file = pool_dir / "directions.json"
    if directions_file.exists():
        directions = json.loads(directions_file.read_text(encoding="utf-8"))
        print(f"  从断点恢复 {len(directions)} 个方向")
    else:
        directions = plan_directions(context_text, n_dirs)
        pool_dir.mkdir(parents=True, exist_ok=True)
        directions_file.write_text(
            json.dumps(directions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"  搜索方向 ({len(directions)} 个):")
    for i, d in enumerate(directions):
        print(f"    [{i}] {d['direction']}: {d['hypothesis'][:60]}...")

    # 3. Round 0 (Original): 每个方向 → 5步循环
    print(f"\n  === Phase 2: Round 0 (Original) ===")
    existing_init = pool.get_by_phase("init")
    existing_init_dirs = {t.direction_id for t in existing_init}

    for d_idx, direction in enumerate(directions):
        if time.time() - t_start > EVOLVED_TOTAL_TIMEOUT:
            print(f"  [超时] 总运行超过 {EVOLVED_TOTAL_TIMEOUT}s")
            break

        if d_idx in existing_init_dirs:
            print(f"\n  --- Direction {d_idx}/{n_dirs} (已有, 跳过) ---")
            continue

        print(f"\n  --- Direction {d_idx}/{n_dirs}: {direction['direction']} ---")
        traj = _run_original_5step(direction, d_idx, pool, dry_run,
                                   factor_pool=factor_pool)
        _try_admit_to_pool(traj, factor_pool, 0)
        pool.save()
        factor_pool.save()

        if traj and traj.failure_step == -1:
            print(f"    结果: ICIR={traj.icir:.4f}, reward={traj.reward:.4f}")
        elif traj:
            print(f"    结果: 失败@step{traj.failure_step}")

    # 4. Rounds 1+ (Mutation/Crossover 交替)
    for round_idx in range(1, n_rounds):
        if time.time() - t_start > EVOLVED_TOTAL_TIMEOUT:
            print(f"  [超时] 总运行超过 {EVOLVED_TOTAL_TIMEOUT}s")
            break

        phase = "mutation" if round_idx % 2 == 1 else "crossover"
        print(f"\n  === Phase {round_idx + 2}: Round {round_idx} ({phase}) ===")

        if phase == "mutation":
            targets = get_mutation_targets(pool, round_idx)
            if not targets:
                print(f"    无可用 mutation 目标, 跳过")
                continue
            print(f"    Mutation 目标: {len(targets)} 个轨迹")
            for parent in targets:
                if time.time() - t_start > EVOLVED_TOTAL_TIMEOUT:
                    break
                print(f"\n    [Mutation] Parent: {parent.id} (ICIR={parent.icir:.4f})")
                traj = mutate_trajectory(parent, round_idx, pool, dry_run=dry_run,
                                         factor_pool=factor_pool)
                _try_admit_to_pool(traj, factor_pool, round_idx)
                pool.save()
                factor_pool.save()
                if traj and traj.failure_step == -1:
                    print(f"    结果: ICIR={traj.icir:.4f}, reward={traj.reward:.4f}")
        else:
            candidates = get_crossover_candidates(pool, round_idx)
            if not candidates:
                print(f"    无可用 crossover 候选, 跳过")
                continue
            groups = select_crossover_groups(candidates)
            print(f"    Crossover 组合: {len(groups)} 组 (from {len(candidates)} 候选)")
            for g_idx, group in enumerate(groups):
                if time.time() - t_start > EVOLVED_TOTAL_TIMEOUT:
                    break
                parent_ids = [p.id for p in group]
                print(f"\n    [Crossover] Group {g_idx}: {parent_ids}")
                traj = crossover_trajectories(group, round_idx, pool, g_idx, dry_run=dry_run,
                                              factor_pool=factor_pool)
                _try_admit_to_pool(traj, factor_pool, round_idx)
                pool.save()
                factor_pool.save()
                if traj and traj.failure_step == -1:
                    print(f"    结果: ICIR={traj.icir:.4f}, reward={traj.reward:.4f}")

    # 5. 从 FactorPool 收集量化准入的因子 (质量筛选)
    print(f"\n  === Phase Final: 收集 + 最终评估 ===")
    all_pool = factor_pool.get_exprs()
    admitted = factor_pool.get_quality_exprs()
    if len(admitted) < len(all_pool):
        filtered_out = len(all_pool) - len(admitted)
        print(f"  质量筛选: {len(all_pool)} 入池 → {len(admitted)} 通过 ({filtered_out} 过滤)")
        # 打印被过滤的因子
        admitted_names = {n for n, _ in admitted}
        for pf in factor_pool.get_all():
            if pf.name not in admitted_names:
                print(f"    过滤: {pf.name} (ICIR={pf.icir:.3f}, RankIC={pf.rank_ic:.4f})")
    stats = pool.stats()
    fp_stats = factor_pool.stats()

    print(f"  轨迹池统计:")
    print(f"    总计: {stats['total']} (成功 {stats['successful']}, 失败 {stats['failed']})")
    print(f"    阶段: init={stats['by_phase']['init']}, "
          f"mutation={stats['by_phase']['mutation']}, "
          f"crossover={stats['by_phase']['crossover']}")
    print(f"    平均 reward: {stats['avg_reward']:.4f}, 最大: {stats['max_reward']:.4f}")
    print(f"  因子池统计:")
    print(f"    入池: {fp_stats['size']}/{fp_stats['capacity']} "
          f"(尝试 {fp_stats['total_attempted']})")
    print(f"    平均|RankIC|: {fp_stats['avg_rank_ic']:.4f}, "
          f"最大: {fp_stats['max_rank_ic']:.4f}")
    print(f"    入池因子: {len(admitted)} 个")

    if admitted:
        print(f"\n  入池因子列表:")
        for name, expr in admitted:
            print(f"    + {name}: {expr[:60]}")

    # 6. 最终 D_expand 评估 + shadow/mined.py
    backtest = None
    if admitted and not dry_run:
        print(f"\n  === 全量 Rolling 回测 ({len(admitted)} 新因子) ===")
        try:
            from factor_lab.mining.backtest_runner import run_comparison
            backtest = run_comparison(admitted)
        except Exception as e:
            print(f"  [backtest] 回测失败: {e}")
            backtest = {"error": str(e)}

    # 7. 保存审计记录
    # 构建 promising factors 列表 (从 pool 中提取)
    promising_factors = []
    for traj in pool.get_successful():
        if traj.best_factor:
            promising_factors.append({
                "name": traj.best_factor['name'],
                "expr": traj.best_factor['expr'],
                "icir": traj.icir,
                "hypothesis": traj.hypothesis[:200],
                "category": traj.direction,
                "phase": traj.phase,
                "is_redundant": False,
                "max_corr": 0,
            })

    run_result = {
        "run_id": run_id,
        "date": today,
        "mode": "evolved" + ("-smoke" if smoke_test else ""),
        "factors_tested": stats['total'],
        "hypotheses_generated": stats['total'],
        "promising_count": len(admitted),
        "promising_factors": promising_factors,
        "non_redundant": [{"name": n, "expr": e} for n, e in admitted],
        "backtest": backtest,
        "pool_stats": stats,
        "directions": [d.get('direction', '') for d in directions],
        "total_time": round(time.time() - t_start, 1),
    }

    run_file = RUNS_DIR / f"{run_id}.json"
    run_file.write_text(
        json.dumps(run_result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\n  审计记录: {run_file}")

    _save_discoveries(promising_factors, run_id)

    # 更新 context
    ctx.update_after_run(run_result)
    print(f"  Agent 记忆已更新")

    # 8. 通知
    beat_baseline = _check_beat_baseline(backtest)
    run_result["beat_baseline"] = beat_baseline

    if beat_baseline and not dry_run:
        _accept_factors_to_mined(admitted, run_id)
        shadow_id = _create_shadow_for_run(run_id, backtest)
        msg = _build_evolved_message(run_result, backtest, stats)
        if shadow_id:
            msg += f"\n\n已自动创建影子验证: {shadow_id} (20交易日)"
        _push_feishu(msg)
        from notifications import write_notification
        write_notification("mining", f"演化因子晋升: #{run_id}", msg)
    elif admitted and not dry_run:
        names = [n for n, _ in admitted[:5]]
        msg = (f"🧬 演化式因子挖掘 #{run_id}\n"
               f"轨迹: {stats['total']} | ACCEPT: {len(admitted)} 因子\n"
               f"因子: {', '.join(names)}\n"
               f"{'回测未超越 M01' if backtest else '未执行回测'}")
        _push_feishu(msg)
        from notifications import write_notification
        write_notification("mining", f"演化非冗余因子: #{run_id}", msg)

    # 打印总结
    print(f"\n{'='*60}")
    print(f"  总结: {stats['total']} 轨迹, "
          f"成功 {stats['successful']}, "
          f"ACCEPT {len(admitted)}")
    if backtest and "baseline" in backtest:
        print(f"  回测: Sharpe {backtest['candidate']['sharpe']:.3f} "
              f"(baseline {backtest['baseline']['sharpe']:.3f}, "
              f"delta {backtest['improvement']['sharpe_delta']:+.3f})")
    print(f"  耗时: {time.time() - t_start:.1f}s")
    print(f"{'='*60}")


def _build_synthetic_parent(d_entry: dict, d_idx: int,
                            global_pool, session_pool) -> "Trajectory | None":
    """从 global_pool 构造 synthetic parent (跨 session mutation)

    前几天探索的方向在 global_pool 中有因子记录，
    但当前 session 的 TrajectoryPool 中没有轨迹。
    从 global_pool 找到该方向最佳因子，构造一个 Trajectory 供 mutation 使用。
    """
    from factor_lab.quanta.trajectory import Trajectory

    direction_name = d_entry.get("direction", "")
    best_factor = None
    best_icir = 0.0

    for pf in global_pool.get_all():
        if pf.direction == direction_name and abs(pf.icir) > abs(best_icir):
            best_factor = pf
            best_icir = pf.icir

    if not best_factor:
        return None

    traj = Trajectory(
        direction_id=d_idx,
        iteration=0,
        phase="init",
        parent_ids=[],
    )
    traj.hypothesis = best_factor.hypothesis
    traj.direction = direction_name
    traj.best_factor = {"name": best_factor.name, "expr": best_factor.expr}
    traj.icir = best_factor.icir
    traj.rank_ic = best_factor.rank_ic
    traj.reward = abs(best_factor.rank_ic)

    # 注册到 session pool 以便 direction_trace 可用
    session_pool.add(traj)
    print(f"    [synthetic] 从 global_pool 构建 parent: "
          f"{best_factor.name} (ICIR={best_factor.icir:.4f})")
    return traj


def _try_admit_to_pool(traj, factor_pool, iteration):
    """尝试将轨迹的最佳因子加入池"""
    if traj and traj.best_factor and traj.failure_step == -1:
        admitted, reason = factor_pool.try_admit(
            name=traj.best_factor['name'],
            expr=traj.best_factor['expr'],
            rank_ic=traj.rank_ic,
            icir=traj.icir,
            hypothesis=traj.hypothesis,
            direction=traj.direction,
            source_traj_id=traj.id,
            iteration=iteration,
        )
        status = "ADMITTED" if admitted else "REJECTED"
        print(f"    因子池: {status} — {reason}")


def _run_original_5step(direction: dict, d_idx: int,
                        pool, dry_run: bool = False,
                        factor_pool=None, experience_memory=None,
                        use_multistage: bool = False):
    """Round 0: 对一个方向执行 5 步循环 (Original)

    复用 evolution.py 的 mutate_trajectory 结构,
    但不需要 parent (这是初始轮)。
    """
    from factor_lab.quanta.trajectory import Trajectory
    from factor_lab.quanta.config import (
        N_CANDIDATES, HISTORY_LIMIT, BACKTEST_ENABLED, ROLLING_EVAL_LITE,
    )
    from factor_lab.quanta import idea_agent, factor_agent, eval_agent

    direction_trace = pool.get_direction_trace(d_idx)

    traj = Trajectory(
        direction_id=d_idx,
        iteration=0,
        phase="init",
        parent_ids=[],
    )

    # --- Step 1: Propose ---
    print(f"    [Step 1] Propose (initial hypothesis)")
    hyp = idea_agent.generate_hypothesis_with_trace(
        direction_trace=direction_trace,
        direction=direction.get('direction', ''),
        suffix="",
        history_limit=HISTORY_LIMIT,
        experience_memory=experience_memory,
    )
    traj.hypothesis = hyp.get('hypothesis', direction.get('hypothesis', ''))
    traj.direction = hyp.get('direction', direction.get('direction', ''))
    traj.mechanism = hyp.get('mechanism', direction.get('mechanism', ''))
    traj.claude_calls += 1

    if dry_run:
        print(f"    [dry-run] 假说: {traj.hypothesis[:80]}...")
        traj.compute_reward()
        pool.add(traj)
        return traj

    # --- Step 2: Construct ---
    print(f"    [Step 2] Construct (with regen loop)")
    trace_text = direction_trace.render_for_prompt()
    factor_list_text = direction_trace.render_factor_list_for_prompt()

    candidates, regen_count = factor_agent.construct_factors_with_regen(
        hypothesis=hyp,
        n=N_CANDIDATES,
        trace_text=trace_text,
        factor_list_text=factor_list_text,
    )
    traj.claude_calls += 1 + regen_count
    traj.regen_attempts = regen_count

    if not candidates:
        traj.failure_step = 1
        traj.error_msg = f"再生循环 {regen_count} 次后仍无有效候选"
        from factor_lab.quanta.evolution import _finalize_trace
        _finalize_trace(traj, pool, direction_trace)
        return traj

    traj.factor_candidates = candidates

    # --- Step 3: Calculate ---
    if use_multistage and factor_pool is not None:
        print(f"    [Step 3] Calculate (multistage pipeline)")
        eval_agent.run_multistage_pipeline(traj, candidates, factor_pool)
    else:
        print(f"    [Step 3] Calculate (validate + evaluate)")
        eval_agent.run_validation_pipeline(traj, candidates)

    # --- Step 4: Backtest ---
    if BACKTEST_ENABLED and traj.best_factor and traj.failure_step == -1:
        use_rolling = ROLLING_EVAL_LITE
        new_factors = [(traj.best_factor['name'], traj.best_factor['expr'])]
        from factor_lab.quanta.evolution import _get_sota_factors
        sota_factors = _get_sota_factors(direction_trace, factor_pool=factor_pool)
        print(f"    [Step 4] Backtest ({'rolling-lite' if use_rolling else 'single-shot'}, "
              f"sota={len(sota_factors)} factors)")
        bt_metrics = eval_agent.run_combined_backtest(new_factors, sota_factors, use_rolling=use_rolling)
        traj.backtest_metrics = bt_metrics
    else:
        print(f"    [Step 4] Backtest (skipped)")

    # --- Step 5: Feedback ---
    from factor_lab.quanta.evolution import _finalize_trace
    _finalize_trace(traj, pool, direction_trace)
    return traj


def _build_evolved_message(run_result: dict, backtest: dict, stats: dict) -> str:
    """构建演化式挖掘的飞书通知"""
    run_id = run_result.get("run_id", "?")
    dt = run_result.get("date", "?")

    lines = [
        f"🧬 演化式因子挖掘 — 发现候选因子",
        "",
        f"运行: #{run_id} ({dt})",
        f"轨迹: {stats.get('total', 0)} | 成功: {stats.get('successful', 0)}",
        f"ACCEPT: {run_result.get('promising_count', 0)} 个因子",
    ]

    # 最佳发现
    best_factors = run_result.get("promising_factors", [])
    if best_factors:
        best_sorted = sorted(best_factors, key=lambda f: abs(f.get('icir', 0)), reverse=True)
        lines.append("")
        lines.append("【最佳发现】")
        for f in best_sorted[:3]:
            lines.append(f"  {f['name']} (ICIR={f.get('icir', 0):.3f}, {f.get('phase', '')})")
            lines.append(f"  {f.get('expr', '')[:80]}")
            lines.append("")

    # 回测对比
    if backtest and "baseline" in backtest:
        bl = backtest["baseline"]
        cd = backtest["candidate"]
        imp = backtest["improvement"]
        lines.append("回测对比 (vs M01-LGB-D3v3r-v2602):")
        lines.append(f"  M01:  Sharpe {bl['sharpe']:.3f} | MDD {bl['mdd']:.2%} | Return {bl['return']:.0%}")
        lines.append(f"  候选: Sharpe {cd['sharpe']:.3f} | MDD {cd['mdd']:.2%} | Return {cd['return']:.0%}")
        mark = "BETTER" if imp["is_better"] else "no improvement"
        lines.append(f"  提升: {imp['sharpe_delta']:+.3f} Sharpe ({mark})")

    lines.append("")
    lines.append(f"接受命令: python -m factor_lab.factor_miner --accept {run_id}")

    return "\n".join(lines)


# ============================================================================
# 每日渐进式挖掘 (--daily)
# ============================================================================

GLOBAL_POOL_FILE = "global_factor_pool.json"
DIRECTION_REGISTRY_FILE = MINING_DIR / "direction_registry.json"


def run_daily_session(smoke_test: bool = False, dry_run: bool = False):
    """每日渐进式挖掘 — 全局 FactorPool + DirectionRegistry

    Phase A: 加载全局状态，判断是否需要规划新方向
    Phase B: 广度 — 探索 pending 方向
    Phase C: 深度 — 对有潜力的方向做 mutation
    Phase D: 质量筛选 → 全量回测 → beat_baseline 则写 mined.py + shadow
    """
    from factor_lab.quanta.config import (
        DAILY_TOTAL_TIMEOUT, DAILY_N_DIRECTIONS, DAILY_BREADTH_STEPS,
        USE_MULTISTAGE,
        IMPORTANCE_SCREEN_ENABLED, IMPORTANCE_MIN_THRESHOLD,
    )
    from factor_lab.quanta.trajectory import TrajectoryPool
    from factor_lab.quanta.factor_pool import FactorPool
    from factor_lab.quanta.evolution import mutate_trajectory
    from factor_lab.quanta.experience_memory import ExperienceMemory
    from factor_lab.mining.planning_agent import plan_directions
    from factor_lab.mining.direction_registry import DirectionRegistry
    from factor_lab.mining.context import MiningContext

    t_start = time.time()
    run_id = _next_run_id()
    today = date.today().isoformat()
    timeout = 3600 if smoke_test else DAILY_TOTAL_TIMEOUT

    print(f"\n{'='*60}")
    print(f"  每日渐进式挖掘 — {run_id} ({today})")
    print(f"  模式: {'smoke-test' if smoke_test else 'daily'}")
    print(f"  时间预算: {timeout/3600:.1f}h")
    print(f"{'='*60}")

    # 初始化 Qlib
    _init_qlib()

    # --- Phase A: 加载全局状态 ---
    print(f"\n  === Phase A: 加载全局状态 ===")

    # 全局 FactorPool (跨天累积)
    global_pool = FactorPool(save_dir=MINING_DIR)
    global_pool.load(GLOBAL_POOL_FILE)
    print(f"  全局因子池: {global_pool.size} 个因子")

    # Experience Memory (FactorMiner 跨 session 经验记忆)
    experience_memory = ExperienceMemory(save_dir=MINING_DIR)
    experience_memory.load()
    mem_text = experience_memory.format_for_prompt()
    if mem_text:
        print(f"  经验记忆: 已加载 ({len(mem_text)} 字符)")
    else:
        print(f"  经验记忆: 空 (首次运行)")

    # DirectionRegistry (方向注册表)
    registry = DirectionRegistry(save_path=DIRECTION_REGISTRY_FILE)
    registry.load()
    reg_stats = registry.stats()
    print(f"  方向注册表: {reg_stats['total']} 个方向 ({reg_stats['by_status']})")

    # 本次 session 的 TrajectoryPool (按 run_id 隔离)
    pool_dir = MINING_DIR / "trajectory_pool" / run_id
    pool = TrajectoryPool(save_dir=pool_dir)

    # 判断是否需要新规划
    ctx = MiningContext()
    context_text = ctx.load()
    n_plan = 2 if smoke_test else DAILY_N_DIRECTIONS

    if registry.needs_new_planning():
        print(f"\n  需要新规划 (pending=0, depth_targets=0)")
        explored_summary = registry.get_explored_summary()
        directions = plan_directions(context_text, n_plan,
                                     explored_summary=explored_summary)
        registry.add_directions(directions, run_id)
        registry.save()
        print(f"  已规划 {len(directions)} 个新方向:")
        for d in directions:
            print(f"    + {d['direction']}: {d['hypothesis'][:60]}...")
    else:
        print(f"  无需新规划")

    # 计算广度/深度比例
    pending = registry.get_pending(limit=100)
    depth_targets = registry.get_depth_targets(limit=100)
    if len(pending) >= 5:
        breadth_ratio = 0.70
    elif len(depth_targets) >= 3:
        breadth_ratio = 0.40
    else:
        breadth_ratio = 0.60

    remaining = timeout - (time.time() - t_start)
    reserve_final = 720  # Phase D 预留 12min (含 importance 训练)
    work_time = max(remaining - reserve_final, 0)
    breadth_time = work_time * breadth_ratio
    depth_time = work_time * (1 - breadth_ratio)
    print(f"\n  时间分配: 广度 {breadth_time/60:.0f}min ({breadth_ratio:.0%}) | "
          f"深度 {depth_time/60:.0f}min ({1-breadth_ratio:.0%}) | "
          f"预留 {reserve_final/60:.0f}min")

    factors_admitted_this_session = 0

    def _admit_and_save(traj, dir_id: str, iteration: int):
        """尝试入池 + 记录结果 + 保存状态 (Phase B/C 共享逻辑)"""
        nonlocal factors_admitted_this_session
        admitted = False
        if traj and traj.best_factor and traj.failure_step == -1:
            ok, reason = global_pool.try_admit(
                name=traj.best_factor['name'],
                expr=traj.best_factor['expr'],
                rank_ic=traj.rank_ic,
                icir=traj.icir,
                hypothesis=traj.hypothesis,
                direction=traj.direction,
                source_traj_id=traj.id,
                iteration=iteration,
            )
            admitted = ok
            print(f"    因子池: {'ADMITTED' if ok else 'REJECTED'} — {reason}")

        registry.record_result(dir_id, admitted=admitted,
                               icir=traj.icir if traj else 0.0,
                               session_id=run_id)
        if admitted:
            factors_admitted_this_session += 1

        pool.save()
        global_pool.save(GLOBAL_POOL_FILE)
        registry.save()

    # --- Phase B: 广度 — 探索 pending 方向 (循环规划) ---
    print(f"\n  === Phase B: 广度探索 ===")
    breadth_deadline = time.time() + breadth_time
    breadth_round = 0

    while time.time() < breadth_deadline:
        pending_dirs = registry.get_pending(limit=20)
        if smoke_test:
            pending_dirs = pending_dirs[:2]

        # 无 pending 方向 → 规划新批次
        if not pending_dirs:
            if smoke_test:
                break  # smoke test 只跑一轮
            remaining_min = (breadth_deadline - time.time()) / 60
            if remaining_min < 5:
                print(f"  [广度] 剩余 {remaining_min:.0f}min, 不再规划")
                break
            breadth_round += 1
            print(f"\n  [广度] 第 {breadth_round+1} 轮规划 (剩余 {remaining_min:.0f}min)")
            explored_summary = registry.get_explored_summary()
            new_dirs = plan_directions(context_text, n_plan,
                                       explored_summary=explored_summary)
            registry.add_directions(new_dirs, run_id)
            registry.save()
            print(f"  已规划 {len(new_dirs)} 个新方向:")
            for d in new_dirs:
                print(f"    + {d['direction']}: {d['hypothesis'][:60]}...")
            continue  # 回到 while 顶部获取新 pending

        if breadth_round == 0:
            print(f"  待探索方向: {len(pending_dirs)} 个")

        for d_entry in pending_dirs:
            if time.time() > breadth_deadline:
                print(f"  [广度超时]")
                break

            dir_id = d_entry["id"]
            print(f"\n  --- [{dir_id}] {d_entry['direction']} ---")
            registry.mark_exploring(dir_id)

            direction = {
                "direction": d_entry["direction"],
                "hypothesis": d_entry["hypothesis"],
                "mechanism": d_entry.get("mechanism", ""),
                "time_scale": d_entry.get("time_scale", ""),
            }

            d_idx = int(dir_id.split("_")[1])
            traj = _run_original_5step(direction, d_idx, pool, dry_run,
                                       factor_pool=global_pool,
                                       experience_memory=experience_memory,
                                       use_multistage=USE_MULTISTAGE)
            _admit_and_save(traj, dir_id, iteration=0)

    # --- Phase C: 深度 — mutation (循环直到超时) ---
    print(f"\n  === Phase C: 深度挖掘 ===")
    depth_deadline = time.time() + depth_time
    depth_round = 0

    while time.time() < depth_deadline:
        depth_dirs = registry.get_depth_targets(limit=10)
        if not depth_dirs:
            depth_dirs = registry.get_explorable(limit=5)
        if smoke_test:
            depth_dirs = depth_dirs[:1]

        if not depth_dirs:
            print(f"  无可 mutation 的方向, 跳过深度阶段")
            break

        if depth_round == 0:
            print(f"  深度目标: {len(depth_dirs)} 个方向")
        else:
            remaining_min = (depth_deadline - time.time()) / 60
            print(f"\n  [深度] 第 {depth_round+1} 轮 ({len(depth_dirs)} 方向, "
                  f"剩余 {remaining_min:.0f}min)")
        depth_round += 1

        any_progress = False
        for d_entry in depth_dirs:
            if time.time() > depth_deadline:
                print(f"  [深度超时]")
                break

            dir_id = d_entry["id"]
            print(f"\n  --- [Mutation] {dir_id}: {d_entry['direction']} "
                  f"(ICIR={d_entry['best_icir']:.3f}) ---")

            d_idx = int(dir_id.split("_")[1])
            direction_trajs = [t for t in pool.get_successful()
                               if t.direction_id == d_idx]

            if not direction_trajs:
                # 前几天的方向: 从 global_pool 构造 synthetic parent
                parent = _build_synthetic_parent(
                    d_entry, d_idx, global_pool, pool)
                if not parent:
                    print(f"    无可用 parent (本session + global_pool均无), 跳过")
                    continue
            else:
                parent = max(direction_trajs, key=lambda t: t.reward)
            print(f"    Parent: {parent.id} (ICIR={parent.icir:.4f})")

            traj = mutate_trajectory(parent, 1, pool, dry_run=dry_run,
                                     factor_pool=global_pool)
            _admit_and_save(traj, dir_id, iteration=depth_round)
            any_progress = True

        if not any_progress:
            print(f"  本轮无进展, 结束深度阶段")
            break

    # --- Phase D: importance 筛选 → 漏斗评估 → mined.py ---
    print(f"\n  === Phase D: 最终评估 (漏斗模式) ===")
    all_pool_factors = global_pool.get_exprs()
    fp_stats = global_pool.stats()
    reg_stats = registry.stats()

    print(f"  全局因子池: {fp_stats['size']} 个因子")
    print(f"  本次新增: {factors_admitted_this_session} 个因子")
    print(f"  方向状态: {reg_stats['by_status']}")

    # Step 1: Importance 筛选 (产出 gain_imp_dict + perm_imp_dict 供漏斗评分使用)
    gain_imp_dict: dict[str, float] = {}
    perm_imp_dict: dict[str, float] = {}
    screened_factors = []
    if (all_pool_factors and factors_admitted_this_session > 0
            and IMPORTANCE_SCREEN_ENABLED and not dry_run):
        try:
            from factor_lab.mining.backtest_runner import screen_by_importance
            screened_factors, gain_imp_dict, perm_imp_dict = screen_by_importance(
                all_pool_factors,
                min_importance=IMPORTANCE_MIN_THRESHOLD,
            )
            # 更新通过筛选因子的 last_validated 时间戳
            screened_names = {n for n, _ in screened_factors}
            global_pool.update_validation(screened_names)
            global_pool.save(GLOBAL_POOL_FILE)
        except Exception as e:
            print(f"  [importance] 筛选失败: {e}, 回退到 IC 质量筛选")
            screened_factors = global_pool.get_quality_exprs()
    elif all_pool_factors and not IMPORTANCE_SCREEN_ENABLED:
        screened_factors = global_pool.get_quality_exprs()
        print(f"  importance 筛选未启用, 回退到 IC 质量筛选: {len(screened_factors)} 因子")
    else:
        screened_factors = global_pool.get_quality_exprs()

    if screened_factors:
        print(f"\n  筛选后因子列表 ({len(screened_factors)}):")
        for name, expr in screened_factors:
            print(f"    + {name}: {expr[:60]}")

    # Step 2: 漏斗评估 (多池并行评分 + Top-N 回测)
    backtest = None
    quality_factors = screened_factors
    funnel_record = None
    if quality_factors and factors_admitted_this_session > 0 and not dry_run:
        t_funnel = time.time()
        try:
            from factor_lab.mining.funnel import FactorFunnel

            # 构建 icir_dict: 从 global_pool 的 PoolFactor 对象
            icir_dict = {f.name: f.icir for f in global_pool.get_all()}

            # 如果 gain_imp_dict 为空 (importance 筛选被跳过), 用 ICIR 作为代理
            if not gain_imp_dict:
                gain_imp_dict = {name: abs(icir) for name, icir in icir_dict.items()}

            funnel = FactorFunnel()
            funnel_pools = funnel.run_funnel(
                all_pool_factors, icir_dict, gain_imp_dict, perm_imp_dict,
            )

            if funnel_pools:
                funnel_record = funnel.to_audit_record(
                    funnel_pools, time.time() - t_funnel)
                print(f"\n{funnel.summary_table(funnel_pools)}")

                # 从回测结果中选最优池
                best = funnel.best_pool(funnel_pools)
                if best and best.backtest and not best.backtest.get("error"):
                    backtest = best.backtest
                    quality_factors = best.factors
                    print(f"\n  最优池: {best.name} ({len(best.factors)} 因子, "
                          f"Sharpe delta={backtest['improvement']['sharpe_delta']:+.3f})")
                else:
                    # 无池通过回测, fallback 到旧逻辑 (单路径)
                    print(f"  [funnel] 无池通过回测, 回退到单路径模式")
                    from factor_lab.mining.backtest_runner import run_comparison
                    backtest = run_comparison(quality_factors)

        except Exception as e:
            print(f"  [funnel] 漏斗评估失败: {e}, 回退到单路径模式")
            try:
                from factor_lab.mining.backtest_runner import run_comparison
                backtest = run_comparison(quality_factors)
            except Exception as e2:
                print(f"  [backtest] 回测失败: {e2}")
                backtest = {"error": str(e2)}

    # 计算 beat_baseline + LLM 统计 (在写审计记录之前)
    beat_baseline = _check_beat_baseline(backtest)

    llm_stats_str = ""
    llm_stats = None
    try:
        from factor_lab.mining.llm_backend import LLMBackend
        llm_stats = LLMBackend.shared().stats()
        if llm_stats.get("usage"):
            usage_parts = [f"{k}={v}" for k, v in llm_stats["usage"].items() if v > 0]
            if usage_parts:
                llm_stats_str = f"\nLLM: {', '.join(usage_parts)}"
    except Exception:
        pass

    # 保存审计记录
    pool_stats = pool.stats()
    run_result = {
        "run_id": run_id,
        "date": today,
        "mode": "daily" + ("-smoke" if smoke_test else ""),
        "factors_tested": pool_stats['total'],
        "hypotheses_generated": pool_stats['total'],
        "promising_count": len(quality_factors),
        "promising_factors": [
            {"name": n, "expr": e} for n, e in quality_factors
        ],
        "non_redundant": [{"name": n, "expr": e} for n, e in quality_factors],
        "backtest": backtest,
        "funnel": funnel_record,
        "beat_baseline": beat_baseline,
        "pool_stats": pool_stats,
        "global_pool_size": fp_stats['size'],
        "direction_stats": reg_stats,
        "factors_admitted_this_session": factors_admitted_this_session,
        "total_time": round(time.time() - t_start, 1),
    }
    if llm_stats_str:  # 仅在有实际 LLM 调用时才记录
        run_result["llm_usage"] = llm_stats

    run_file = RUNS_DIR / f"{run_id}.json"
    run_file.write_text(
        json.dumps(run_result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\n  审计记录: {run_file}")

    # 更新 context
    ctx.update_after_run(run_result)

    # Experience Memory 演化 + 保存
    all_trajs = pool.all()
    if all_trajs:
        experience_memory.evolve(all_trajs, global_pool)
        experience_memory.save()
        print(f"  经验记忆已更新 (从 {len(all_trajs)} 条轨迹提炼)")

    # 通知 + mined.py + shadow
    if beat_baseline and not dry_run:
        _accept_factors_to_mined(quality_factors, run_id)
        shadow_id = _create_shadow_for_run(run_id, backtest)
        msg = _build_daily_message(run_result, backtest, reg_stats)
        if funnel_record:
            best_name = funnel_record.get("best_pool", "?")
            msg += f"\n\n漏斗最优池: {best_name}"
        if shadow_id:
            msg += f"\n已自动创建影子验证: {shadow_id} (20交易日)"
        msg += llm_stats_str
        _push_feishu(msg)
        from notifications import write_notification
        write_notification("mining", f"每日挖掘晋升: #{run_id}", msg)
    elif factors_admitted_this_session > 0 and not dry_run:
        msg = (f"📅 每日因子挖掘 #{run_id}\n"
               f"新增: {factors_admitted_this_session} 因子 | "
               f"全局池: {fp_stats['size']}\n"
               f"{'回测未超越 M01' if backtest else '未执行回测 (无新质量因子)'}"
               f"{llm_stats_str}")
        _push_feishu(msg)
        from notifications import write_notification
        write_notification("mining", f"每日挖掘: #{run_id}", msg)

    # 打印总结
    print(f"\n{'='*60}")
    print(f"  总结: {pool_stats['total']} 轨迹, "
          f"新增 {factors_admitted_this_session} 因子, "
          f"全局池 {fp_stats['size']}")
    if backtest and "baseline" in backtest:
        print(f"  回测: Sharpe {backtest['candidate']['sharpe']:.3f} "
              f"(baseline {backtest['baseline']['sharpe']:.3f}, "
              f"delta {backtest['improvement']['sharpe_delta']:+.3f})")
    if funnel_record:
        print(f"  漏斗: {len(funnel_record['pools'])} 池评估, "
              f"最优={funnel_record.get('best_pool', 'N/A')}")
    print(f"  耗时: {time.time() - t_start:.1f}s")
    print(f"{'='*60}")


def _build_daily_message(run_result: dict, backtest: dict, reg_stats: dict) -> str:
    """构建每日挖掘的飞书通知"""
    run_id = run_result.get("run_id", "?")
    dt = run_result.get("date", "?")
    new_count = run_result.get("factors_admitted_this_session", 0)
    pool_size = run_result.get("global_pool_size", 0)

    status_str = ", ".join(f"{k}={v}" for k, v in reg_stats.get("by_status", {}).items())
    lines = [
        f"📅 每日因子挖掘 — 发现候选因子",
        "",
        f"运行: #{run_id} ({dt})",
        f"新增因子: {new_count} | 全局池: {pool_size}",
        f"方向: {reg_stats.get('total', 0)} 个 ({status_str})",
    ]

    if backtest and "baseline" in backtest:
        bl = backtest["baseline"]
        cd = backtest["candidate"]
        imp = backtest["improvement"]
        lines.append("")
        lines.append("回测对比 (vs M01-LGB-D3v3r-v2602):")
        lines.append(f"  M01:  Sharpe {bl['sharpe']:.3f} | MDD {bl['mdd']:.2%} | Return {bl['return']:.0%}")
        lines.append(f"  候选: Sharpe {cd['sharpe']:.3f} | MDD {cd['mdd']:.2%} | Return {cd['return']:.0%}")
        mark = "BETTER" if imp["is_better"] else "no improvement"
        lines.append(f"  提升: {imp['sharpe_delta']:+.3f} Sharpe ({mark})")

    lines.append("")
    lines.append(f"接受命令: python -m factor_lab.factor_miner --accept {run_id}")
    return "\n".join(lines)


def main():
    try:
        multiprocessing.set_start_method('fork', force=True)
    except (ValueError, RuntimeError):
        pass  # Windows 无 fork，使用默认 spawn

    parser = argparse.ArgumentParser(description="自动因子挖掘")
    # 模式选择 (互斥)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--daily", action="store_true",
                            help="每日渐进式挖掘 (默认)")
    mode_group.add_argument("--evolved", action="store_true",
                            help="演化式挖掘 (单次全量)")
    mode_group.add_argument("--legacy", action="store_true",
                            help="旧流程 (4轮种子/mutation/crossover)")
    # 通用参数
    parser.add_argument("--dry-run", action="store_true", help="不推送不回测")
    parser.add_argument("--skip-backtest", action="store_true", help="跳过回测")
    parser.add_argument("--smoke-test", action="store_true",
                        help="快速验证 (3方向x2轮, 约35min)")
    parser.add_argument("--report", action="store_true", help="查看历史报告")
    parser.add_argument("--accept", type=str, help="接受某次运行的因子 (如 run_001)")
    args = parser.parse_args()

    if args.report:
        show_report()
        return

    if args.accept:
        accept_run(args.accept)
        return

    _ensure_dirs()

    if not _acquire_lock():
        print("另一个挖掘进程正在运行，退出")
        return

    try:
        if args.legacy:
            run_mining(dry_run=args.dry_run,
                       skip_backtest=args.skip_backtest or args.dry_run)
        elif args.evolved:
            run_evolved_mining(dry_run=args.dry_run,
                               smoke_test=args.smoke_test)
        else:
            # 默认: 每日渐进式挖掘
            run_daily_session(smoke_test=args.smoke_test,
                              dry_run=args.dry_run)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
