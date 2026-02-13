#!/usr/bin/env python3
"""自动因子挖掘 — Claude Agent 驱动，每周运行

用法:
    python -m factor_lab.factor_miner              # 完整周期
    python -m factor_lab.factor_miner --dry-run    # 不推送不回测
    python -m factor_lab.factor_miner --report     # 查看历史
    python -m factor_lab.factor_miner --accept run_001  # 人工接受因子
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

MINING_DIR = PROJECT_DIR / "factor_lab" / "mining_results"
RUNS_DIR = MINING_DIR / "runs"
DISCOVERIES_DIR = MINING_DIR / "discoveries"
LOCK_FILE = MINING_DIR / ".lock"

MAX_CYCLES = 3  # 最多迭代轮数
BATCH_SIZE = 10  # 每轮生成因子数
TOTAL_TIMEOUT = 3600  # 总运行超时 (秒)


def _ensure_dirs():
    for d in [MINING_DIR, RUNS_DIR, DISCOVERIES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _acquire_lock() -> bool:
    """lockfile 防并发"""
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)  # 检查进程是否存活
            return False
        except (ProcessLookupError, ValueError, PermissionError):
            pass  # 旧锁，进程已死或 PID 被复用
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
    import qlib
    from qlib.constant import REG_CN
    try:
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_bs", region=REG_CN)
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
        user_id = os.environ.get("FEISHU_USER_OPEN_ID", "")

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
    from factor_lab.mining.hypothesis import generate_hypotheses
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

    # 2. 迭代生成和评估
    for cycle in range(MAX_CYCLES):
        if time.time() - t_start > TOTAL_TIMEOUT:
            print(f"  [超时] 总运行超过 {TOTAL_TIMEOUT}s")
            break

        print(f"\n  --- 第 {cycle + 1}/{MAX_CYCLES} 轮 ---")

        # 生成假说
        hypotheses = generate_hypotheses(
            ctx.load(), existing, batch_size=BATCH_SIZE, focus=focus,
        )
        print(f"  生成假说: {len(hypotheses)} 个")
        all_hypotheses.extend(hypotheses)

        if not hypotheses:
            failed_approaches.append(f"cycle_{cycle+1}: Claude 未返回有效假说")
            focus = "之前方向无效，请尝试完全不同的因子类型"
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
            focus = "之前表达式有语法错误，请更仔细构造"
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
            focus = "之前因子预测力不足，请尝试更强的信号因子"
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
            promising_factors.append({
                "name": n,
                "expr": e,
                "icir": icir,
                "hypothesis": h_info.get("hypothesis", ""),
                "category": h_info.get("category", ""),
                "is_redundant": redundancy.get(n, {}).get("is_redundant", False),
                "max_corr": redundancy.get(n, {}).get("max_corr", 0),
            })

        non_redundant_factors.extend(non_redundant)
        if non_redundant:
            print(f"  非冗余 promising 因子: {len(non_redundant)} 个")
            break  # 有发现，进入回测

        failed_approaches.append(f"cycle_{cycle+1}: promising 因子全部冗余")
        focus = "之前因子与已有因子高相关，请设计更独特的因子"

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
    run_file.write_text(json.dumps(run_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  审计记录: {run_file}")

    # 保存 discovery
    for pf in promising_factors:
        if not pf.get("is_redundant"):
            disc_file = DISCOVERIES_DIR / f"{run_id}_{pf['name']}.json"
            disc_file.write_text(json.dumps(pf, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5. 更新 context
    ctx.update_after_run(run_result)
    print(f"  Agent 记忆已更新")

    # 6. 通知
    beat_baseline = backtest and isinstance(backtest, dict) and backtest.get("improvement", {}).get("is_better")
    run_result["beat_baseline"] = beat_baseline

    if beat_baseline and not dry_run:
        # 写入 mined.py + 自动创建 shadow 验证
        _accept_factors_to_mined(non_redundant_factors, run_id)
        shadow_id = _create_shadow_for_run(run_id, backtest)
        msg = _build_discovery_message(run_result, backtest)
        if shadow_id:
            msg += f"\n\n已自动创建影子验证: {shadow_id} (20交易日)"
        _push_feishu(msg)
    elif promising_factors and not dry_run:
        # 有 promising 但未 beat baseline，也发一条简要通知
        names = [f["name"] for f in promising_factors if not f.get("is_redundant")]
        if names:
            msg = (f"🔬 因子挖掘 #{run_id}\n"
                   f"发现 {len(names)} 个非冗余因子: {', '.join(names[:5])}\n"
                   f"{'回测未超越 M01' if backtest else '未执行回测'}")
            _push_feishu(msg)

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
        idx = content.rfind("]")
        content = content[:idx] + "\n" + insert_text + "\n" + content[idx:]

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


def main():
    multiprocessing.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser(description="自动因子挖掘")
    parser.add_argument("--dry-run", action="store_true", help="不推送不回测")
    parser.add_argument("--skip-backtest", action="store_true", help="跳过回测")
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
        run_mining(dry_run=args.dry_run, skip_backtest=args.skip_backtest or args.dry_run)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
