"""Paper-style plotting utilities for saved sensitivity experiment results."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats


PAPER_COLORS = {
    "condition_met": "#228B22",
    "condition_not_met": "#B22222",
    "empirical_snr": "#4169E1",
    "predicted_snr": "#FF8C00",
    "gcn_accuracy": "#228B22",
    "fnn_baseline": "#8A2BE2",
    "edge": "#c7c7c7",
}


NODE_COLUMNS = [
    "backbone",
    "backbone_type",
    "homophily_h",
    "run_idx",
    "node_idx",
    "label",
    "degree",
    "gcn_sensitivity_condition_met",
    "gcn_snr_theorem_node",
    "gcn_snr_mc_node",
    "fnn_snr_theorem_node",
    "fnn_snr_mc_node",
    "gcn_accuracy_node",
    "fnn_accuracy_node",
    "gcn_bottleneck_score_node",
]

GRAPH_COLUMNS = [
    "backbone",
    "backbone_type",
    "homophily_h",
    "run_idx",
    "n_layers",
    "gcn_avg_snr_theorem",
    "gcn_avg_snr_mc",
    "fnn_avg_snr_theorem",
    "gcn_test_accuracy_graph",
    "fnn_test_accuracy_graph",
    "gcn_higher_order_homophily_graph",
]


def _read_csv_with_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    usecols = [column for column in columns if column in header]
    return pd.read_csv(path, usecols=usecols)


def _find_result_csv(results_dir: Path, kind: str) -> Path:
    matches = sorted(results_dir.glob(f"{kind}_results_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {kind}_results_*.csv file found in {results_dir}")
    return matches[-1]


def load_sensitivity_results(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load node and graph CSVs from an existing sensitivity result directory."""
    results_dir = Path(results_dir)
    graph_df = _read_csv_with_columns(_find_result_csv(results_dir, "graph"), GRAPH_COLUMNS)
    node_df = _read_csv_with_columns(_find_result_csv(results_dir, "node"), NODE_COLUMNS)

    if "backbone" not in graph_df.columns:
        graph_df["backbone"] = "vanilla_gcn"
    if "backbone" not in node_df.columns:
        node_df["backbone"] = "vanilla_gcn"
    return node_df, graph_df


def _as_bool_float(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    if series.dtype == object:
        return series.map({"True": True, "False": False, True: True, False: False}).astype(float)
    return series.astype(float)


def _ci95(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= 1:
        return 0.0
    return float(1.96 * series.sem())


def _nearest_value(values: pd.Series, target: float) -> float:
    unique = np.asarray(sorted(pd.to_numeric(values.dropna().unique(), errors="coerce")))
    if unique.size == 0:
        raise ValueError("No finite homophily values are available")
    return float(unique[np.argmin(np.abs(unique - target))])


def _backbones(df: pd.DataFrame) -> list[str]:
    if "backbone" not in df.columns:
        return ["vanilla_gcn"]
    return sorted(df["backbone"].dropna().astype(str).unique())


def average_node_iterations(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average node-level results over training/feature-sampling iterations.

    The raw CSV has one row per run. Paper node-level plots use per-node
    quantities after averaging over `run_idx` for each `(backbone, h, node)`.
    """
    df = node_df.copy()
    if "backbone" not in df.columns:
        df["backbone"] = "vanilla_gcn"

    for column in ["gcn_accuracy_node", "fnn_accuracy_node", "gcn_sensitivity_condition_met"]:
        df[column] = _as_bool_float(df[column])

    group_cols = ["backbone", "homophily_h", "node_idx"]
    grouped = df.groupby(group_cols, as_index=False).agg(
        label=("label", "first"),
        degree=("degree", "first"),
        gcn_accuracy_node=("gcn_accuracy_node", "mean"),
        fnn_accuracy_node=("fnn_accuracy_node", "mean"),
        gcn_sensitivity_condition_rate=("gcn_sensitivity_condition_met", "mean"),
        gcn_snr_theorem_node=("gcn_snr_theorem_node", "mean"),
        gcn_snr_mc_node=("gcn_snr_mc_node", "mean"),
        fnn_snr_theorem_node=("fnn_snr_theorem_node", "mean"),
        fnn_snr_mc_node=("fnn_snr_mc_node", "mean"),
        gcn_bottleneck_score_node=("gcn_bottleneck_score_node", "mean"),
        n_runs=("run_idx", "nunique"),
    )
    grouped["gcn_sensitivity_condition_met"] = grouped["gcn_sensitivity_condition_rate"] > 0.5
    grouped["accuracy_improvement"] = grouped["gcn_accuracy_node"] - grouped["fnn_accuracy_node"]
    grouped["accuracy_improved"] = grouped["accuracy_improvement"] > 0
    grouped["snr_improvement_empirical_gcn_theorem_fnn"] = (
        grouped["gcn_snr_mc_node"] - grouped["fnn_snr_theorem_node"]
    )
    grouped["snr_improvement_theorem"] = grouped["gcn_snr_theorem_node"] - grouped["fnn_snr_theorem_node"]
    return grouped


def average_graph_iterations(graph_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average graph-level results over `run_idx` for each backbone and homophily.
    """
    df = graph_df.copy()
    if "backbone" not in df.columns:
        df["backbone"] = "vanilla_gcn"

    group_cols = ["backbone", "homophily_h"]
    return df.groupby(group_cols, as_index=False).agg(
        n_layers=("n_layers", "first") if "n_layers" in df.columns else ("homophily_h", "size"),
        n_runs=("run_idx", "nunique"),
        gcn_acc_mean=("gcn_test_accuracy_graph", "mean"),
        gcn_acc_ci=("gcn_test_accuracy_graph", _ci95),
        fnn_acc_mean=("fnn_test_accuracy_graph", "mean"),
        fnn_acc_ci=("fnn_test_accuracy_graph", _ci95),
        gcn_snr_mc_mean=("gcn_avg_snr_mc", "mean"),
        gcn_snr_mc_ci=("gcn_avg_snr_mc", _ci95),
        gcn_snr_theorem_mean=("gcn_avg_snr_theorem", "mean"),
        gcn_snr_theorem_ci=("gcn_avg_snr_theorem", _ci95),
        fnn_snr_theorem_mean=("fnn_avg_snr_theorem", "mean"),
        fnn_snr_theorem_ci=("fnn_avg_snr_theorem", _ci95),
        higher_order_homophily_mean=("gcn_higher_order_homophily_graph", "mean"),
        higher_order_homophily_ci=("gcn_higher_order_homophily_graph", _ci95),
    )


def _load_saved_graph(results_dir: Path, h_value: float):
    graph_dir = results_dir / "graphs"
    if not graph_dir.exists():
        return None

    candidates = []
    for path in graph_dir.glob("graph_h_*.pt"):
        match = re.search(r"graph_h_([0-9.]+)_", path.name)
        if match:
            candidates.append((abs(float(match.group(1)) - h_value), path))
    if not candidates:
        return None

    graph_path = sorted(candidates, key=lambda item: item[0])[0][1]
    import torch

    try:
        graph_data = torch.load(graph_path, weights_only=False)
    except TypeError:
        graph_data = torch.load(graph_path)
    if isinstance(graph_data, dict) and "graph" in graph_data:
        return graph_data["graph"]
    return graph_data


def _draw_graph_panel(ax, graph, positions, values, title, palette):
    graph_nx = graph.to_networkx().to_undirected()
    nx.draw_networkx_edges(graph_nx, positions, ax=ax, edge_color=PAPER_COLORS["edge"], width=0.35, alpha=0.35)
    node_order = list(graph_nx.nodes())
    colors = [palette[values.loc[node]] for node in node_order]
    ax.scatter(
        [positions[node][0] for node in node_order],
        [positions[node][1] for node in node_order],
        c=colors,
        s=18,
        linewidths=0.15,
        edgecolors="#222222",
        zorder=3,
    )
    ax.set_title(title)
    ax.set_axis_off()


def plot_paper_graph_predictions(
    node_df: pd.DataFrame,
    save_path: str | Path,
    results_dir: str | Path | None = None,
    h_value: float = 0.5,
    backbone: str = "vanilla_gcn",
):
    """Figure-2a style node-level condition and accuracy-improvement plot."""
    avg_node = average_node_iterations(node_df)
    h_to_plot = _nearest_value(avg_node["homophily_h"], h_value)
    plot_data = avg_node[
        (avg_node["homophily_h"] == h_to_plot) & (avg_node["backbone"] == backbone)
    ].copy()
    if plot_data.empty:
        raise ValueError(f"No node rows found for backbone={backbone!r}, h={h_to_plot:.3f}")
    plot_data = plot_data.sort_values("node_idx").set_index("node_idx")

    graph = _load_saved_graph(Path(results_dir), h_to_plot) if results_dir is not None else None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
    if graph is not None:
        graph = graph.cpu()
        graph_nx = graph.to_networkx().to_undirected()
        positions = nx.spring_layout(graph_nx, seed=8, iterations=80)
        _draw_graph_panel(
            axes[0],
            graph,
            positions,
            plot_data["gcn_sensitivity_condition_met"],
            f"Sensitivity condition, h={h_to_plot:.2f}",
            {True: PAPER_COLORS["condition_met"], False: PAPER_COLORS["condition_not_met"]},
        )
        _draw_graph_panel(
            axes[1],
            graph,
            positions,
            plot_data["accuracy_improved"],
            "GCN accuracy exceeds FNN",
            {True: PAPER_COLORS["condition_met"], False: PAPER_COLORS["condition_not_met"]},
        )
    else:
        axes[0].scatter(
            plot_data.index,
            plot_data["degree"],
            c=np.where(plot_data["gcn_sensitivity_condition_met"], PAPER_COLORS["condition_met"], PAPER_COLORS["condition_not_met"]),
            s=12,
        )
        axes[1].scatter(
            plot_data.index,
            plot_data["degree"],
            c=np.where(plot_data["accuracy_improved"], PAPER_COLORS["condition_met"], PAPER_COLORS["condition_not_met"]),
            s=12,
        )
        axes[0].set_title(f"Sensitivity condition, h={h_to_plot:.2f}")
        axes[1].set_title("GCN accuracy exceeds FNN")
        for ax in axes:
            ax.set_xlabel("Node index")
            ax.set_ylabel("Degree")

    fig.suptitle(f"{backbone}: node-level prediction of message-passing benefit", y=1.03)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paper_snr_vs_accuracy(
    node_df: pd.DataFrame,
    graph_df: pd.DataFrame,
    save_path: str | Path,
    backbone: str = "vanilla_gcn",
    log_x: bool = False,
):
    """Figure-2c style node-level SNR/accuracy scatter with marginals."""
    avg_node = average_node_iterations(node_df)
    node_data = avg_node[avg_node["backbone"] == backbone].copy()
    if node_data.empty:
        raise ValueError(f"No node rows found for backbone={backbone!r}")

    summary = average_graph_iterations(graph_df)
    summary = summary[summary["backbone"] == backbone].sort_values("homophily_h")

    finite = np.isfinite(node_data["gcn_snr_mc_node"]) & np.isfinite(node_data["gcn_accuracy_node"])
    if log_x:
        finite &= node_data["gcn_snr_mc_node"] > 0
    node_data = node_data[finite]
    if node_data.empty:
        scale_label = "positive " if log_x else ""
        raise ValueError(f"No {scale_label}finite node rows found for backbone={backbone!r}")
    condition_met = node_data[node_data["gcn_sensitivity_condition_met"]]
    condition_not_met = node_data[~node_data["gcn_sensitivity_condition_met"]]

    fig = plt.figure(figsize=(8.0, 6.5))
    gs = GridSpec(2, 2, width_ratios=[4.0, 1.05], height_ratios=[1.0, 4.0], hspace=0.05, wspace=0.06)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(
        condition_not_met["gcn_snr_mc_node"],
        condition_not_met["gcn_accuracy_node"],
        s=13,
        alpha=0.45,
        color=PAPER_COLORS["condition_not_met"],
        linewidths=0,
        label="Condition not met",
    )
    ax_main.scatter(
        condition_met["gcn_snr_mc_node"],
        condition_met["gcn_accuracy_node"],
        s=13,
        alpha=0.45,
        color=PAPER_COLORS["condition_met"],
        linewidths=0,
        label="Condition met",
    )

    if not summary.empty:
        empirical_summary = summary
        predicted_summary = summary
        if log_x:
            empirical_summary = empirical_summary[empirical_summary["gcn_snr_mc_mean"] > 0]
            predicted_summary = predicted_summary[predicted_summary["gcn_snr_theorem_mean"] > 0]

        ax_main.errorbar(
            empirical_summary["gcn_snr_mc_mean"],
            empirical_summary["gcn_acc_mean"],
            xerr=empirical_summary["gcn_snr_mc_ci"],
            yerr=empirical_summary["gcn_acc_ci"],
            fmt="^",
            color=PAPER_COLORS["empirical_snr"],
            ecolor=PAPER_COLORS["empirical_snr"],
            capsize=2,
            markersize=5,
            label="Empirical graph averages",
            zorder=4,
        )
        ax_main.errorbar(
            predicted_summary["gcn_snr_theorem_mean"],
            predicted_summary["gcn_acc_mean"],
            xerr=predicted_summary["gcn_snr_theorem_ci"],
            yerr=predicted_summary["gcn_acc_ci"],
            fmt="s",
            color=PAPER_COLORS["predicted_snr"],
            ecolor=PAPER_COLORS["predicted_snr"],
            capsize=2,
            markersize=4,
            label="Predicted graph averages",
            zorder=5,
        )

    fnn_acc = node_data["fnn_accuracy_node"].mean()
    fnn_snr = node_data["fnn_snr_theorem_node"].mean()
    ax_main.axhline(fnn_acc, color=PAPER_COLORS["fnn_baseline"], linestyle="--", linewidth=1.5, label="Mean FNN accuracy")
    ax_main.axvline(fnn_snr, color=PAPER_COLORS["fnn_baseline"], linestyle="--", linewidth=1.5, label="Mean theorem FNN SNR")

    x_values = node_data["gcn_snr_mc_node"].to_numpy()
    y_values = node_data["gcn_accuracy_node"].to_numpy()
    x_min, x_max = np.nanpercentile(x_values, [0.5, 99.5])
    y_min, y_max = np.nanpercentile(y_values, [0.5, 99.5])
    y_pad = max((y_max - y_min) * 0.08, 0.03)
    if log_x:
        log_x_min, log_x_max = np.log10([x_min, x_max])
        log_x_pad = max((log_x_max - log_x_min) * 0.08, 0.05)
        ax_main.set_xlim(10 ** (log_x_min - log_x_pad), 10 ** (log_x_max + log_x_pad))
        ax_main.set_xscale("log")
        ax_top.set_xscale("log")
    else:
        x_pad = max((x_max - x_min) * 0.08, 1e-6)
        ax_main.set_xlim(max(0, x_min - x_pad), x_max + x_pad)
    ax_main.set_ylim(max(0, y_min - y_pad), min(1.03, y_max + y_pad))

    bins_x = np.geomspace(*ax_main.get_xlim(), 45) if log_x else np.linspace(*ax_main.get_xlim(), 45)
    bins_y = np.linspace(*ax_main.get_ylim(), 35)
    ax_top.hist(condition_not_met["gcn_snr_mc_node"], bins=bins_x, color=PAPER_COLORS["condition_not_met"], alpha=0.55, density=True)
    ax_top.hist(condition_met["gcn_snr_mc_node"], bins=bins_x, color=PAPER_COLORS["condition_met"], alpha=0.55, density=True)
    ax_right.hist(condition_not_met["gcn_accuracy_node"], bins=bins_y, color=PAPER_COLORS["condition_not_met"], alpha=0.55, density=True, orientation="horizontal")
    ax_right.hist(condition_met["gcn_accuracy_node"], bins=bins_y, color=PAPER_COLORS["condition_met"], alpha=0.55, density=True, orientation="horizontal")
    ax_right.axhline(fnn_acc, color=PAPER_COLORS["fnn_baseline"], linestyle="--", linewidth=1.2)
    ax_top.axvline(fnn_snr, color=PAPER_COLORS["fnn_baseline"], linestyle="--", linewidth=1.2)

    ax_main.set_xlabel("Node-level empirical SNR" + (" (log scale)" if log_x else ""))
    ax_main.set_ylabel("Node-level GCN accuracy")
    ax_top.set_ylabel("Density")
    ax_right.set_xlabel("Density")
    ax_main.grid(alpha=0.25)
    ax_main.legend(loc="lower right", fontsize=8, frameon=True)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    ax_top.spines[["right", "top"]].set_visible(False)
    ax_right.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"{backbone}: SNR predicts node-level accuracy", y=0.98)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paper_fnn_snr_empirical_vs_theorem(node_df: pd.DataFrame, save_path: str | Path):
    """Plot empirical FNN SNR against theorem-predicted FNN SNR per node."""
    avg_node = average_node_iterations(node_df)
    backbones = _backbones(avg_node)
    fig, axes = plt.subplots(
        1,
        len(backbones),
        figsize=(4.35 * len(backbones), 4.15),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    scatter = None

    for col, backbone in enumerate(backbones):
        ax = axes[0, col]
        data = avg_node[avg_node["backbone"] == backbone].copy()
        data["fnn_snr_theorem_node"] = pd.to_numeric(data["fnn_snr_theorem_node"], errors="coerce")
        data["fnn_snr_mc_node"] = pd.to_numeric(data["fnn_snr_mc_node"], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["homophily_h", "fnn_snr_theorem_node", "fnn_snr_mc_node"]
        )

        scatter = ax.scatter(
            data["fnn_snr_theorem_node"],
            data["fnn_snr_mc_node"],
            c=data["homophily_h"],
            cmap="viridis",
            s=8,
            alpha=0.22,
            linewidths=0,
            rasterized=True,
        )

        values = pd.concat([data["fnn_snr_theorem_node"], data["fnn_snr_mc_node"]])
        if not values.empty:
            lower, upper = np.nanpercentile(values, [0.5, 99.5])
            pad = max((upper - lower) * 0.08, 1e-5)
            lower = max(0.0, lower - pad)
            upper = upper + pad
            ax.plot([lower, upper], [lower, upper], color="#333333", linestyle="--", linewidth=1.1, label="y=x")
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)

        ax.set_title(backbone)
        ax.set_xlabel("Theorem-predicted FNN SNR")
        if col == 0:
            ax.set_ylabel("Empirical FNN SNR")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Edge homophily h", fraction=0.035, pad=0.02)
    fig.suptitle("Empirical vs theorem-predicted FNN SNR per node", y=0.96)
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.17, top=0.82, wspace=0.24)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paper_snr_homophily_summary(graph_df: pd.DataFrame, save_path: str | Path):
    """Graph-wide SNR and accuracy as a function of SBM edge homophily."""
    summary = average_graph_iterations(graph_df)
    backbones = _backbones(summary)
    fig, axes = plt.subplots(len(backbones), 1, figsize=(7.2, 3.0 * len(backbones)), sharex=True, squeeze=False)

    for row, backbone in enumerate(backbones):
        ax_acc = axes[row, 0]
        ax_snr = ax_acc.twinx()
        data = summary[summary["backbone"] == backbone].sort_values("homophily_h")
        h = data["homophily_h"]

        ax_acc.errorbar(h, data["gcn_acc_mean"], yerr=data["gcn_acc_ci"], color=PAPER_COLORS["gcn_accuracy"], marker="o", capsize=2, label="GCN accuracy")
        ax_acc.errorbar(h, data["fnn_acc_mean"], yerr=data["fnn_acc_ci"], color=PAPER_COLORS["fnn_baseline"], marker="x", linestyle="--", capsize=2, label="FNN accuracy")
        ax_snr.errorbar(h, data["gcn_snr_mc_mean"], yerr=data["gcn_snr_mc_ci"], color=PAPER_COLORS["empirical_snr"], marker="^", capsize=2, label="Empirical SNR")
        ax_snr.errorbar(h, data["gcn_snr_theorem_mean"], yerr=data["gcn_snr_theorem_ci"], color=PAPER_COLORS["predicted_snr"], marker="s", linestyle="--", capsize=2, label="Predicted SNR")

        ax_acc.set_ylabel("Accuracy")
        ax_snr.set_ylabel("SNR")
        ax_acc.set_title(backbone)
        ax_acc.grid(alpha=0.25)
        lines_1, labels_1 = ax_acc.get_legend_handles_labels()
        lines_2, labels_2 = ax_snr.get_legend_handles_labels()
        ax_acc.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=8)

    axes[-1, 0].set_xlabel("Edge homophily h")
    fig.suptitle("Graph-averaged SNR and accuracy over homophily", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paper_bottlenecking_snr(node_df: pd.DataFrame, save_path: str | Path):
    """Figure-3a style bottlenecking-score/SNR validation, faceted by backbone."""
    avg_node = average_node_iterations(node_df)
    backbones = _backbones(avg_node)
    fig, axes = plt.subplots(1, len(backbones), figsize=(4.8 * len(backbones), 4.1), sharey=True, squeeze=False)

    for col, backbone in enumerate(backbones):
        ax = axes[0, col]
        data = avg_node[avg_node["backbone"] == backbone]
        scatter = ax.scatter(
            data["gcn_bottleneck_score_node"],
            data["gcn_snr_mc_node"],
            c=data["gcn_accuracy_node"],
            cmap="viridis",
            s=10,
            alpha=0.45,
            linewidths=0,
        )
        valid = data[["gcn_bottleneck_score_node", "gcn_snr_mc_node"]].dropna()
        if len(valid) > 2 and valid["gcn_bottleneck_score_node"].nunique() > 1:
            corr, p_value = scipy_stats.pearsonr(valid["gcn_bottleneck_score_node"], valid["gcn_snr_mc_node"])
            ax.set_title(f"{backbone}\nr={corr:.2f}, p={p_value:.1e}")
        else:
            ax.set_title(backbone)
        ax.set_xlabel("Class-bottlenecking score")
        if col == 0:
            ax.set_ylabel("Node-level empirical SNR")
        ax.grid(alpha=0.22)
        fig.colorbar(scatter, ax=ax, label="GCN accuracy")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value == 0:
        return "<1e-300"
    if p_value < 1e-3:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def plot_paper_condition_improvement_boxplots(node_df: pd.DataFrame, save_path: str | Path):
    """Boxplots of accuracy and SNR improvement grouped by sensitivity-condition pass/fail."""
    avg_node = average_node_iterations(node_df).copy()
    avg_node["condition"] = np.where(avg_node["gcn_sensitivity_condition_met"], "Pass", "Fail")
    condition_order = ["Fail", "Pass"]
    condition_palette = {
        "Fail": PAPER_COLORS["condition_not_met"],
        "Pass": PAPER_COLORS["condition_met"],
    }
    metrics = [
        ("accuracy_improvement", "Accuracy improvement\n(GCN - FNN)"),
        (
            "snr_improvement_empirical_gcn_theorem_fnn",
            "SNR improvement\n(empirical GCN - theorem FNN)",
        ),
    ]
    backbones = _backbones(avg_node)
    fig, axes = plt.subplots(
        len(backbones),
        len(metrics),
        figsize=(7.4, 2.75 * len(backbones)),
        sharex=True,
        squeeze=False,
    )

    for row, backbone in enumerate(backbones):
        backbone_data = avg_node[avg_node["backbone"] == backbone].copy()
        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            plot_data = backbone_data[["condition", metric]].replace([np.inf, -np.inf], np.nan).dropna()
            sns.boxplot(
                data=plot_data,
                x="condition",
                hue="condition",
                y=metric,
                order=condition_order,
                hue_order=condition_order,
                palette=condition_palette,
                width=0.58,
                linewidth=1.1,
                fliersize=1.8,
                legend=False,
                ax=ax,
            )
            ax.axhline(0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.75)

            passed = plot_data.loc[plot_data["condition"] == "Pass", metric]
            failed = plot_data.loc[plot_data["condition"] == "Fail", metric]
            if len(passed) > 1 and len(failed) > 1:
                p_value = scipy_stats.ttest_ind(passed, failed, equal_var=False, nan_policy="omit").pvalue
                title = f"{backbone}\nWelch p={_format_p_value(p_value)}"
            else:
                title = f"{backbone}\nWelch p=n/a"
            ax.set_title(title)
            ax.set_xlabel("Sensitivity condition")
            ax.set_ylabel(ylabel if col == 0 else ylabel)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Node-level improvements by sensitivity-condition outcome", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paper_higher_order_homophily(graph_df: pd.DataFrame, save_path: str | Path, mean_degree: float | None = None):
    """Empirical higher-order homophily against the planted-partition SBM approximation."""
    if "gcn_higher_order_homophily_graph" not in graph_df.columns:
        return

    df = graph_df.copy()
    if "n_layers" not in df.columns:
        df["n_layers"] = 1
    dedup_cols = ["homophily_h", "run_idx", "n_layers", "gcn_higher_order_homophily_graph"]
    empirical = df[dedup_cols].drop_duplicates()
    summary = empirical.groupby(["homophily_h", "n_layers"], as_index=False).agg(
        mean=("gcn_higher_order_homophily_graph", "mean"),
        ci=("gcn_higher_order_homophily_graph", _ci95),
        n_runs=("run_idx", "nunique"),
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for n_layers, data in summary.groupby("n_layers"):
        data = data.sort_values("homophily_h")
        order = int(2 * n_layers)
        h = data["homophily_h"].to_numpy()
        theory = 0.5 + 0.5 * np.power(2 * h - 1, order)
        ax.errorbar(h, data["mean"], yerr=data["ci"], fmt="o", capsize=2, label=f"Empirical order {order}")
        ax.plot(h, theory, linestyle="--", label=f"SBM approximation order {order}")
        if mean_degree:
            band = 1.0 / mean_degree
            ax.fill_between(h, np.clip(theory - band, 0, None), np.clip(theory + band, None, 1), alpha=0.12)

    ax.set_xlabel("Edge homophily h")
    ax.set_ylabel("Higher-order homophily")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_paper_sensitivity_plots_from_frames(
    node_df: pd.DataFrame,
    graph_df: pd.DataFrame,
    output_dir: str | Path,
    results_dir: str | Path | None = None,
    primary_backbone: str = "vanilla_gcn",
    h_value: float = 0.5,
    mean_degree: float | None = None,
    snr_log_x: bool = False,
) -> list[Path]:
    """Generate all paper-style sensitivity plots from already loaded frames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir) if results_dir is not None else output_dir

    available_backbones = _backbones(node_df)
    if primary_backbone not in available_backbones:
        primary_backbone = available_backbones[0]

    sns.set_theme(style="whitegrid", context="paper")
    written: list[Path] = []

    path = output_dir / f"paper_graph_predictions_{primary_backbone}_h{h_value:.2f}.png"
    plot_paper_graph_predictions(node_df, path, results_dir=results_dir, h_value=h_value, backbone=primary_backbone)
    written.append(path)

    for backbone in available_backbones:
        suffix = "_logx" if snr_log_x else ""
        path = output_dir / f"paper_snr_vs_acc_scatter_plus_histograms_{backbone}{suffix}.png"
        plot_paper_snr_vs_accuracy(node_df, graph_df, path, backbone=backbone, log_x=snr_log_x)
        written.append(path)

    path = output_dir / "paper_fnn_snr_empirical_vs_theorem_per_node.png"
    plot_paper_fnn_snr_empirical_vs_theorem(node_df, path)
    written.append(path)

    path = output_dir / "paper_graph_wide_snr_accuracy_by_homophily.png"
    plot_paper_snr_homophily_summary(graph_df, path)
    written.append(path)

    path = output_dir / "paper_bottlenecking_snr_scatter.png"
    plot_paper_bottlenecking_snr(node_df, path)
    written.append(path)

    path = output_dir / "paper_condition_improvement_boxplots.png"
    plot_paper_condition_improvement_boxplots(node_df, path)
    written.append(path)

    path = output_dir / "paper_higher_order_homophily_sbm.png"
    plot_paper_higher_order_homophily(graph_df, path, mean_degree=mean_degree)
    if path.exists():
        written.append(path)

    return written


def generate_paper_sensitivity_plots(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    primary_backbone: str = "vanilla_gcn",
    h_value: float = 0.5,
    mean_degree: float | None = None,
    snr_log_x: bool = False,
) -> list[Path]:
    """Generate paper-style plots directly from an existing result directory."""
    results_dir = Path(results_dir)
    node_df, graph_df = load_sensitivity_results(results_dir)
    if output_dir is None:
        output_dir = results_dir / "paper_plots"
    return generate_paper_sensitivity_plots_from_frames(
        node_df,
        graph_df,
        output_dir,
        results_dir=results_dir,
        primary_backbone=primary_backbone,
        h_value=h_value,
        mean_degree=mean_degree,
        snr_log_x=snr_log_x,
    )
