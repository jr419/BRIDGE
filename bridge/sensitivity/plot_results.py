"""CLI for regenerating sensitivity plots from saved CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .paper_plots import generate_paper_sensitivity_plots


def _mean_degree_from_config(results_dir: Path) -> float | None:
    config_files = sorted(results_dir.glob("config_*.yaml"))
    if not config_files:
        return None
    with config_files[-1].open("r") as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("sbm_params", {}).get("mean_degree")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style sensitivity plots from existing result CSVs."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing node_results_*.csv and graph_results_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots. Defaults to <results-dir>/paper_plots.",
    )
    parser.add_argument(
        "--primary-backbone",
        default="vanilla_gcn",
        help="Backbone used for the graph-panel plot.",
    )
    parser.add_argument(
        "--h-value",
        type=float,
        default=0.5,
        help="Homophily value for the node-level graph-panel plot.",
    )
    parser.add_argument(
        "--mean-degree",
        type=float,
        default=None,
        help="Mean degree used for the SBM approximation band. Defaults to the saved config value.",
    )
    parser.add_argument(
        "--snr-log-x",
        action="store_true",
        help="Use a log-scaled x-axis for SNR-vs-accuracy scatter plots.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    mean_degree = args.mean_degree
    if mean_degree is None:
        mean_degree = _mean_degree_from_config(results_dir)

    written = generate_paper_sensitivity_plots(
        results_dir=results_dir,
        output_dir=args.output_dir,
        primary_backbone=args.primary_backbone,
        h_value=args.h_value,
        mean_degree=mean_degree,
        snr_log_x=args.snr_log_x,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
