#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import ttest_rel, wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


ORDER = ["fedavg", "random", "greedy", "heuristic", "dqn", "ddqn", "tl_ppo"]
DISPLAY = {
    "fedavg": "FedAvg",
    "random": "Random",
    "greedy": "Greedy",
    "heuristic": "Heuristic",
    "dqn": "DQN",
    "ddqn": "DDQN",
    "tl_ppo": "Telemetry-Driven PPO",
}

PPO_NAME = "Telemetry-Driven PPO"


def load_results(results_dir: str) -> pd.DataFrame:
    rows = []
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json"):
            continue
        fpath = os.path.join(results_dir, fn)
        try:
            with open(fpath, "r") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] Skipping unreadable JSON: {fpath} ({e})")
            continue

        if "history" not in obj:
            print(f"[WARN] Skipping file without 'history': {fpath}")
            continue

        inferred_seed = obj.get("seed", None)

        for h in obj["history"]:
            row = dict(h)
            row["_source_file"] = fn
            if "seed" not in row and inferred_seed is not None:
                row["seed"] = inferred_seed
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid result JSON files found in: {results_dir}")

    df = pd.DataFrame(rows)

    required_cols = ["round", "strategy", "global_acc", "reward", "latency_s"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in results data: {missing}")

    # numeric conversions
    numeric_cols = [
        "round", "global_acc", "reward", "latency_s", "seed",
        "comm_bytes", "uplink_bytes", "downlink_bytes",
        "model_bytes", "telemetry_bytes"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["round", "strategy", "global_acc", "reward", "latency_s"]).copy()
    df["round"] = df["round"].astype(int)

    if "seed" not in df.columns:
        df["seed"] = df["_source_file"]

    # standardize strategy names
    df["strategy"] = df["strategy"].astype(str).str.strip().str.lower()

    return df


def get_summary_window(df: pd.DataFrame, tail_window: int) -> Tuple[int, int]:
    max_round = int(df["round"].max())
    min_round = int(df["round"].min())
    start_round = max(min_round, max_round - tail_window + 1)
    end_round = max_round
    return start_round, end_round


def get_plot_df(df: pd.DataFrame, plot_start_round: Optional[int]) -> pd.DataFrame:
    if plot_start_round is None:
        return df.copy()
    return df[df["round"] >= plot_start_round].copy()


def maybe_compute_comm_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a unified 'comm_bytes_total' column if possible.
    Priority:
      1) existing comm_bytes
      2) uplink_bytes + downlink_bytes
      3) model_bytes + telemetry_bytes
    """
    df = df.copy()

    if "comm_bytes" in df.columns:
        df["comm_bytes_total"] = df["comm_bytes"].fillna(0.0)
        return df

    if "uplink_bytes" in df.columns or "downlink_bytes" in df.columns:
        up = df["uplink_bytes"] if "uplink_bytes" in df.columns else 0.0
        down = df["downlink_bytes"] if "downlink_bytes" in df.columns else 0.0
        df["comm_bytes_total"] = pd.to_numeric(up, errors="coerce").fillna(0.0) + pd.to_numeric(down, errors="coerce").fillna(0.0)
        return df

    if "model_bytes" in df.columns or "telemetry_bytes" in df.columns:
        mb = df["model_bytes"] if "model_bytes" in df.columns else 0.0
        tb = df["telemetry_bytes"] if "telemetry_bytes" in df.columns else 0.0
        df["comm_bytes_total"] = pd.to_numeric(mb, errors="coerce").fillna(0.0) + pd.to_numeric(tb, errors="coerce").fillna(0.0)
        return df

    return df


def curve_plot(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(8, 5))
    plotted_any = False

    for s in ORDER:
        d = df[df["strategy"] == s].copy()
        if d.empty or metric not in d.columns:
            continue

        grp = d.groupby("round")[metric]
        mean = grp.mean().sort_index()
        std = grp.std().fillna(0.0).sort_index()

        if mean.empty:
            continue

        plotted_any = True
        plt.plot(mean.index, mean.values, label=DISPLAY.get(s, s))
        plt.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.2)

    if not plotted_any:
        print(f"[WARN] No data available for plot: {title}")
        plt.close()
        return

    plt.xlabel("Federated Communication Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] {out_path}")


def final_bar_plot(
    tail_stats: pd.DataFrame,
    metric_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    out_path: str,
    lower_is_better: bool = False,
):
    plot_rows = []
    for s in ORDER:
        name = DISPLAY.get(s, s)
        d = tail_stats[tail_stats["Strategy"] == name]
        if d.empty or metric_col not in d.columns or std_col not in d.columns:
            continue
        plot_rows.append(d.iloc[0])

    if not plot_rows:
        print(f"[WARN] No rows available for final bar plot: {title}")
        return

    pdf = pd.DataFrame(plot_rows)

    plt.figure(figsize=(8, 5))
    x = list(range(len(pdf)))
    y = pdf[metric_col].tolist()
    yerr = pdf[std_col].tolist()

    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xticks(x, pdf["Strategy"].tolist(), rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] {out_path}")


def aggregate_tail_per_seed(df: pd.DataFrame, summary_start: int, summary_end: int) -> pd.DataFrame:
    tail = df[df["round"].between(summary_start, summary_end)].copy()
    if tail.empty:
        raise RuntimeError("Tail window has no data")

    agg_spec = {
        "global_acc": "mean",
        "reward": "mean",
        "latency_s": "mean",
    }

    if "comm_bytes_total" in tail.columns:
        agg_spec["comm_bytes_total"] = "mean"

    tail_seed = tail.groupby(["strategy", "seed"], as_index=False).agg(agg_spec)
    return tail_seed


def build_table_ii(
    df: pd.DataFrame,
    out_csv: str,
    summary_start: int,
    summary_end: int,
) -> pd.DataFrame:
    tail_seed = aggregate_tail_per_seed(df, summary_start, summary_end)

    rows = []
    for s in ORDER:
        d = tail_seed[tail_seed["strategy"] == s].copy()
        if d.empty:
            continue

        acc_mean = d["global_acc"].mean()
        acc_std = d["global_acc"].std(ddof=1) if len(d) > 1 else 0.0
        reward_mean = d["reward"].mean()
        reward_std = d["reward"].std(ddof=1) if len(d) > 1 else 0.0
        lat_mean = d["latency_s"].mean()
        lat_std = d["latency_s"].std(ddof=1) if len(d) > 1 else 0.0

        row = {
            "Strategy": DISPLAY.get(s, s),
            "Accuracy (mean ± std) ↑": f"{acc_mean:.4f} ± {acc_std:.4f}",
            "Reward (mean ± std) ↑": f"{reward_mean:.4f} ± {reward_std:.4f}",
            "Latency (s) (mean ± std) ↓": f"{lat_mean:.4f} ± {lat_std:.4f}",
            "_acc_mean": acc_mean,
            "_acc_std": acc_std,
            "_reward_mean": reward_mean,
            "_reward_std": reward_std,
            "_lat_mean": lat_mean,
            "_lat_std": lat_std,
        }

        if "comm_bytes_total" in d.columns:
            comm_mean = d["comm_bytes_total"].mean()
            comm_std = d["comm_bytes_total"].std(ddof=1) if len(d) > 1 else 0.0
            row["Communication Cost (bytes) (mean ± std) ↓"] = f"{comm_mean:.2f} ± {comm_std:.2f}"
            row["_comm_mean"] = comm_mean
            row["_comm_std"] = comm_std

        rows.append(row)

    if not rows:
        raise RuntimeError("No rows available to build Table II")

    t2 = pd.DataFrame(rows)

    # dominant markers
    best_acc = t2["_acc_mean"].max()
    best_reward = t2["_reward_mean"].max()
    best_lat = t2["_lat_mean"].min()
    best_comm = t2["_comm_mean"].min() if "_comm_mean" in t2.columns else None

    t2["Best Accuracy?"] = t2["_acc_mean"].apply(lambda x: "BEST" if math.isclose(x, best_acc, rel_tol=1e-12, abs_tol=1e-12) else "")
    t2["Best Reward?"] = t2["_reward_mean"].apply(lambda x: "BEST" if math.isclose(x, best_reward, rel_tol=1e-12, abs_tol=1e-12) else "")
    t2["Best Latency?"] = t2["_lat_mean"].apply(lambda x: "BEST" if math.isclose(x, best_lat, rel_tol=1e-12, abs_tol=1e-12) else "")
    if best_comm is not None:
        t2["Best Communication Cost?"] = t2["_comm_mean"].apply(lambda x: "BEST" if math.isclose(x, best_comm, rel_tol=1e-12, abs_tol=1e-12) else "")

    t2.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")
    return t2


def build_table_iii(t2: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    if "Strategy" not in t2.columns:
        raise RuntimeError("Table II missing 'Strategy' column")

    if PPO_NAME not in t2["Strategy"].values:
        raise RuntimeError(f"{PPO_NAME} not found in Table II")

    baselines = t2[t2["Strategy"] != PPO_NAME].copy()
    if baselines.empty:
        raise RuntimeError("No baseline strategies found for Table III")

    ppo = t2[t2["Strategy"] == PPO_NAME].iloc[0]

    best_acc = baselines.sort_values("_acc_mean", ascending=False).iloc[0]
    best_reward = baselines.sort_values("_reward_mean", ascending=False).iloc[0]
    best_lat = baselines.sort_values("_lat_mean", ascending=True).iloc[0]

    rows = [
        {
            "Metric": "Accuracy",
            "Best Baseline": best_acc["Strategy"],
            "Best Baseline Value": f"{best_acc['_acc_mean']:.4f}",
            "Telemetry-Driven PPO": f"{ppo['_acc_mean']:.4f}",
            "Improvement": f"{100.0 * (ppo['_acc_mean'] - best_acc['_acc_mean']) / max(abs(best_acc['_acc_mean']), 1e-8):.2f}%",
            "Note": "Higher is better",
        },
        {
            "Metric": "Reward",
            "Best Baseline": best_reward["Strategy"],
            "Best Baseline Value": f"{best_reward['_reward_mean']:.4f}",
            "Telemetry-Driven PPO": f"{ppo['_reward_mean']:.4f}",
            "Improvement": f"{ppo['_reward_mean'] - best_reward['_reward_mean']:.4f}",
            "Note": "Absolute difference used; percentage is misleading near zero",
        },
        {
            "Metric": "Latency",
            "Best Baseline": best_lat["Strategy"],
            "Best Baseline Value": f"{best_lat['_lat_mean']:.4f}",
            "Telemetry-Driven PPO": f"{ppo['_lat_mean']:.4f}",
            "Improvement": f"{100.0 * (best_lat['_lat_mean'] - ppo['_lat_mean']) / max(abs(best_lat['_lat_mean']), 1e-8):.2f}%",
            "Note": "Lower is better",
        },
    ]

    if "_comm_mean" in t2.columns:
        best_comm = baselines.sort_values("_comm_mean", ascending=True).iloc[0]
        rows.append(
            {
                "Metric": "Communication Cost",
                "Best Baseline": best_comm["Strategy"],
                "Best Baseline Value": f"{best_comm['_comm_mean']:.2f}",
                "Telemetry-Driven PPO": f"{ppo['_comm_mean']:.2f}",
                "Improvement": f"{100.0 * (best_comm['_comm_mean'] - ppo['_comm_mean']) / max(abs(best_comm['_comm_mean']), 1e-8):.2f}%",
                "Note": "Lower is better",
            }
        )

    t3 = pd.DataFrame(rows)
    t3.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")
    return t3


def build_window_sensitivity_table(df: pd.DataFrame, out_csv: str, windows: List[int]) -> pd.DataFrame:
    rows = []
    max_round = int(df["round"].max())

    for w in windows:
        start_round = max(int(df["round"].min()), max_round - w + 1)
        end_round = max_round
        tail_seed = aggregate_tail_per_seed(df, start_round, end_round)

        for s in ORDER:
            d = tail_seed[tail_seed["strategy"] == s].copy()
            if d.empty:
                continue

            row = {
                "tail_window": w,
                "Strategy": DISPLAY.get(s, s),
                "acc_mean": d["global_acc"].mean(),
                "acc_std": d["global_acc"].std(ddof=1) if len(d) > 1 else 0.0,
                "reward_mean": d["reward"].mean(),
                "reward_std": d["reward"].std(ddof=1) if len(d) > 1 else 0.0,
                "lat_mean": d["latency_s"].mean(),
                "lat_std": d["latency_s"].std(ddof=1) if len(d) > 1 else 0.0,
            }
            if "comm_bytes_total" in d.columns:
                row["comm_mean"] = d["comm_bytes_total"].mean()
                row["comm_std"] = d["comm_bytes_total"].std(ddof=1) if len(d) > 1 else 0.0

            rows.append(row)

    ws = pd.DataFrame(rows)
    ws.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")
    return ws


def compare_significance(
    df: pd.DataFrame,
    summary_start: int,
    summary_end: int,
    out_csv: str,
    baseline_strategy: Optional[str] = None,
):
    if not SCIPY_AVAILABLE:
        print("[WARN] scipy not available; skipping significance tests")
        return

    tail_seed = aggregate_tail_per_seed(df, summary_start, summary_end)

    ppo = tail_seed[tail_seed["strategy"] == "tl_ppo"].copy()
    if ppo.empty:
        print("[WARN] tl_ppo not found; skipping significance tests")
        return

    baselines = [baseline_strategy] if baseline_strategy else [s for s in ORDER if s != "tl_ppo"]

    rows = []
    for b in baselines:
        bdf = tail_seed[tail_seed["strategy"] == b].copy()
        if bdf.empty:
            continue

        merged = pd.merge(
            ppo[["seed", "global_acc", "reward", "latency_s"]],
            bdf[["seed", "global_acc", "reward", "latency_s"]],
            on="seed",
            suffixes=("_ppo", "_base"),
        )
        if len(merged) < 2:
            continue

        for metric in ["global_acc", "reward", "latency_s"]:
            x = merged[f"{metric}_ppo"]
            y = merged[f"{metric}_base"]

            try:
                stat, p_t = ttest_rel(x, y)
            except Exception:
                p_t = None

            try:
                _, p_w = wilcoxon(x, y)
            except Exception:
                p_w = None

            rows.append(
                {
                    "metric": metric,
                    "baseline": DISPLAY.get(b, b),
                    "n_paired_seeds": len(merged),
                    "ppo_mean": x.mean(),
                    "baseline_mean": y.mean(),
                    "paired_t_pvalue": p_t,
                    "wilcoxon_pvalue": p_w,
                }
            )

    if rows:
        sig = pd.DataFrame(rows)
        sig.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv}")
    else:
        print("[WARN] No significance-test rows produced")


def write_summary_info(
    outdir: str,
    df: pd.DataFrame,
    summary_start: int,
    summary_end: int,
    tail_window: int,
    plot_start_round: Optional[int],
):
    info_path = os.path.join(outdir, "summary_window.txt")
    max_round = int(df["round"].max())
    min_round = int(df["round"].min())
    strategies = sorted(df["strategy"].dropna().unique().tolist())
    seeds = sorted(df["seed"].dropna().unique().tolist()) if "seed" in df.columns else []

    with open(info_path, "w") as f:
        f.write(f"Available rounds: {min_round}-{max_round}\n")
        f.write(f"Plot rounds used: {plot_start_round if plot_start_round is not None else min_round}-{max_round}\n")
        f.write(f"Tail window size: {tail_window}\n")
        f.write(f"Table summary rounds used: {summary_start}-{summary_end}\n")
        f.write(f"Strategies found: {', '.join(strategies)}\n")
        if seeds:
            f.write(f"Seeds found: {', '.join(str(x) for x in seeds)}\n")
        if "comm_bytes_total" in df.columns:
            f.write("Communication-cost column available: comm_bytes_total\n")
        else:
            f.write("Communication-cost column available: NO\n")

    print(f"[OK] {info_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tail_window", type=int, default=20, help="Tail window for Table II / III, e.g. 20, 50, 100, 150")
    ap.add_argument("--plot_start_round", type=int, default=None, help="Optional starting round for plots, e.g. 200")
    ap.add_argument("--sensitivity_windows", type=int, nargs="*", default=[20, 50, 100, 150], help="Tail windows for sensitivity table")
    ap.add_argument("--run_significance", action="store_true", help="Run paired significance tests against baselines")
    ap.add_argument("--significance_baseline", type=str, default=None, help="Optional single baseline strategy key, e.g. random or dqn")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_results(args.results_dir)
    df = maybe_compute_comm_cost(df)

    summary_start, summary_end = get_summary_window(df, args.tail_window)
    plot_df = get_plot_df(df, args.plot_start_round)

    print(f"[INFO] Plot rounds used: {int(plot_df['round'].min())}-{int(plot_df['round'].max())}")
    print(f"[INFO] Table summary rounds used: {summary_start}-{summary_end}")

    # Curves
    curve_plot(
        plot_df,
        metric="global_acc",
        ylabel="Accuracy",
        title="Accuracy convergence across federated learning rounds",
        out_path=os.path.join(args.outdir, "Figure3_accuracy_convergence.png"),
    )

    curve_plot(
        plot_df,
        metric="reward",
        ylabel="Reward",
        title="Evolution of reinforcement learning reward",
        out_path=os.path.join(args.outdir, "Figure4_reward_evolution.png"),
    )

    curve_plot(
        plot_df,
        metric="latency_s",
        ylabel="Latency (s)",
        title="Comparison of end-to-end training latency under different offloading strategies",
        out_path=os.path.join(args.outdir, "Figure5_latency_comparison.png"),
    )

    if "comm_bytes_total" in plot_df.columns:
        curve_plot(
            plot_df,
            metric="comm_bytes_total",
            ylabel="Communication Cost (bytes)",
            title="Communication cost across federated learning rounds",
            out_path=os.path.join(args.outdir, "Figure6_communication_cost.png"),
        )

    # Tables
    t2 = build_table_ii(
        df=df,
        out_csv=os.path.join(args.outdir, "TableII_performance_comparison.csv"),
        summary_start=summary_start,
        summary_end=summary_end,
    )

    t3 = build_table_iii(
        t2=t2,
        out_csv=os.path.join(args.outdir, "TableIII_relative_improvement.csv"),
    )

    # Final bar charts
    final_bar_plot(
        t2,
        metric_col="_acc_mean",
        std_col="_acc_std",
        ylabel="Accuracy",
        title=f"Final accuracy comparison over last {args.tail_window} rounds",
        out_path=os.path.join(args.outdir, "Figure_bar_accuracy_tail.png"),
    )

    final_bar_plot(
        t2,
        metric_col="_lat_mean",
        std_col="_lat_std",
        ylabel="Latency (s)",
        title=f"Final latency comparison over last {args.tail_window} rounds",
        out_path=os.path.join(args.outdir, "Figure_bar_latency_tail.png"),
    )

    if "_comm_mean" in t2.columns:
        final_bar_plot(
            t2,
            metric_col="_comm_mean",
            std_col="_comm_std",
            ylabel="Communication Cost (bytes)",
            title=f"Final communication-cost comparison over last {args.tail_window} rounds",
            out_path=os.path.join(args.outdir, "Figure_bar_comm_tail.png"),
        )

    # sensitivity
    build_window_sensitivity_table(
        df=df,
        out_csv=os.path.join(args.outdir, "Table_window_sensitivity.csv"),
        windows=args.sensitivity_windows,
    )

    # optional significance
    if args.run_significance:
        compare_significance(
            df=df,
            summary_start=summary_start,
            summary_end=summary_end,
            out_csv=os.path.join(args.outdir, "Table_significance_tests.csv"),
            baseline_strategy=args.significance_baseline,
        )

    write_summary_info(
        outdir=args.outdir,
        df=df,
        summary_start=summary_start,
        summary_end=summary_end,
        tail_window=args.tail_window,
        plot_start_round=args.plot_start_round,
    )


if __name__ == "__main__":
    main()
