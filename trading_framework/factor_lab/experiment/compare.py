"""实验结果对比 — 比较不同因子预设和模型的回测效果"""
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiments"


def load_all_results() -> pd.DataFrame:
    """加载所有实验结果"""
    results = []
    if not RESULTS_DIR.exists():
        return pd.DataFrame()

    for f in RESULTS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                results.append(data)
        except Exception:
            continue

    return pd.DataFrame(results) if results else pd.DataFrame()


def compare_presets(model: str = "lightgbm") -> pd.DataFrame:
    """固定模型，比较不同因子预设"""
    df = load_all_results()
    if len(df) == 0:
        print("无实验结果")
        return df

    df = df[df["model"] == model]
    if len(df) == 0:
        print(f"无 {model} 的结果")
        return df

    cols = ["preset", "model", "total_return", "annual_return", "sharpe",
            "max_drawdown", "excess_return", "train_time"]
    available = [c for c in cols if c in df.columns]
    df = df[available].sort_values("sharpe", ascending=False)

    print(f"\n{'='*80}")
    print(f"因子预设对比 (模型: {model})")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    return df


def compare_models(preset: str = "alpha158_ext") -> pd.DataFrame:
    """固定因子预设，比较不同模型"""
    df = load_all_results()
    if len(df) == 0:
        print("无实验结果")
        return df

    df = df[df["preset"] == preset]
    if len(df) == 0:
        print(f"无 {preset} 的结果")
        return df

    cols = ["preset", "model", "total_return", "annual_return", "sharpe",
            "max_drawdown", "excess_return", "train_time"]
    available = [c for c in cols if c in df.columns]
    df = df[available].sort_values("sharpe", ascending=False)

    print(f"\n{'='*80}")
    print(f"模型对比 (因子: {preset})")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    return df


def full_comparison() -> pd.DataFrame:
    """完整对比矩阵"""
    df = load_all_results()
    if len(df) == 0:
        print("无实验结果")
        return df

    # 透视表: preset × model → sharpe
    if "sharpe" in df.columns and "error" not in df.columns:
        valid = df[df["sharpe"].notna()]
        if len(valid) > 0:
            pivot = valid.pivot_table(
                values="sharpe", index="preset", columns="model", aggfunc="first"
            )
            print(f"\n{'='*80}")
            print("Sharpe 对比矩阵 (preset × model)")
            print(f"{'='*80}")
            print(pivot.to_string(float_format="%.3f"))
            return pivot

    return df
