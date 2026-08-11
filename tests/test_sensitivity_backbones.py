from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from bridge.sensitivity import create_sensitivity_model, estimate_sensitivity_autograd
from bridge.sensitivity.run_experiment import run_full_sensitivity_experiment


BACKBONES = [
    "vanilla_gcn",
    "initial_residual_gcn",
    "gcnii",
    "h2gcn",
]


@pytest.mark.parametrize("backbone_type", BACKBONES)
def test_sensitivity_backbone_forward_shapes_and_sensitivity(tiny_graph, backbone_type):
    g = tiny_graph(n=6, undirected=True, k=2)
    labels = g.ndata["label"]
    model = create_sensitivity_model(
        backbone_type,
        in_feats=5,
        hidden_feats=4,
        out_feats=2,
        n_layers=2,
        alpha=0.1,
        lambda_=0.5,
    ).double()

    x_2d = torch.randn(g.num_nodes(), 5, dtype=torch.double)
    out_2d = model(g, x_2d)
    assert out_2d.shape == (g.num_nodes(), 2)
    assert torch.isfinite(out_2d).all()

    x_3d = torch.randn(g.num_nodes(), 5, 3, dtype=torch.double)
    out_3d = model(g, x_3d)
    assert out_3d.shape == (g.num_nodes(), 2, 3)
    assert torch.isfinite(out_3d).all()

    sensitivity = estimate_sensitivity_autograd(
        model,
        g,
        in_feats=5,
        labels=labels,
        sensitivity_type="noise",
        device="cpu",
    )
    assert sensitivity.shape == (g.num_nodes(), 2, 5, 5)
    assert torch.isfinite(sensitivity).all()


def test_vanila_gcn_alias(tiny_graph):
    g = tiny_graph(n=4, undirected=True, k=2)
    model = create_sensitivity_model(
        "vanila_gcn",
        in_feats=3,
        hidden_feats=4,
        out_feats=2,
        n_layers=1,
    ).double()
    x = torch.randn(g.num_nodes(), 3, dtype=torch.double)
    assert model(g, x).shape == (g.num_nodes(), 2)


@pytest.mark.parametrize("alias", ["H2GCN", "h2_gcn", "h2-gcn"])
def test_h2gcn_aliases(tiny_graph, alias):
    g = tiny_graph(n=4, undirected=True, k=2)
    model = create_sensitivity_model(
        alias,
        in_feats=3,
        hidden_feats=4,
        out_feats=2,
        n_layers=1,
    ).double()
    x = torch.randn(g.num_nodes(), 3, dtype=torch.double)
    assert model(g, x).shape == (g.num_nodes(), 2)


def test_full_sensitivity_experiment_expands_rows_by_backbone(tmp_path, monkeypatch):
    config_path = Path("config_examples/sensitivity_smoke.yaml")
    config = yaml.safe_load(config_path.read_text())
    config["results_dir"] = str(tmp_path / "sensitivity_smoke")
    config["model_params"]["n_epochs"] = 1

    # The runner smoke focuses on model/backbone execution and CSV schema.
    # Plot behavior is covered by the same result columns but skipped here for speed.
    import bridge.sensitivity.visualization as visualization

    monkeypatch.setattr(visualization, "plot_local_sensitivity_validation", lambda *args, **kwargs: None)
    monkeypatch.setattr(visualization, "plot_snr_ratio_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(visualization, "plot_node_acc_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(visualization, "plot_bottlenecking_snr_scatter", lambda *args, **kwargs: None)
    monkeypatch.setattr(visualization, "plot_graph_wide_snr_validation", lambda *args, **kwargs: None)

    test_config_path = tmp_path / "sensitivity_smoke.yaml"
    test_config_path.write_text(yaml.safe_dump(config))

    results_dir = run_full_sensitivity_experiment(test_config_path)
    graph_df = pd.read_csv(next(results_dir.glob("graph_results_*.csv")))
    node_df = pd.read_csv(next(results_dir.glob("node_results_*.csv")))

    expected_backbones = {"vanilla_gcn", "initial_residual_gcn", "gcnii", "h2gcn"}
    assert set(graph_df["backbone"]) == expected_backbones
    assert set(node_df["backbone"]) == expected_backbones

    expected_graph_rows = (
        config["sbm_params"]["h_steps"]
        * config["experiment_params"]["num_total_runs"]
        * len(expected_backbones)
    )
    expected_node_rows = expected_graph_rows * config["sbm_params"]["n_nodes"]
    assert len(graph_df) == expected_graph_rows
    assert len(node_df) == expected_node_rows

    for column in ["backbone_type", "alpha", "lambda", "gcn_avg_snr_mc", "gcn_test_accuracy_graph"]:
        assert column in graph_df.columns
