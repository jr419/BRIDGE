import numpy as np
import torch

import dgl

from bridge.rewiring.pipeline import (
    run_bridge_pipeline,
    run_bridge_experiment,
    run_iterative_bridge_pipeline,
    run_iterative_bridge_experiment,
)


def test_run_bridge_pipeline_smoke(tiny_graph):
    g = tiny_graph(n=6, undirected=True, k=3)
    P_k = np.eye(3)
    out = run_bridge_pipeline(
        g,
        P_k=P_k,
        h_feats_mpnn=8,
        n_layers_mpnn=1,
        dropout_p_mpnn=0.0,
        model_lr_mpnn=1e-2,
        wd_mpnn=0.0,
        n_epochs=1,
        early_stopping=1,
        d_out=5,
        num_graphs=1,
        device='cpu',
        seed=0,
        log_training=False,
        dataset_name='cora',
        do_self_loop=False,
    )
    # basic structure
    assert {'cold_start', 'rewired', 'original_stats', 'rewired_stats'}.issubset(out.keys())


def test_run_bridge_experiment_smoke(tiny_graph):
    g = tiny_graph(n=6, undirected=True, k=3)
    P_k = np.eye(3)
    stats, results_list = run_bridge_experiment(
        g,
        P_k=P_k,
        h_feats_mpnn=8,
        n_layers_mpnn=1,
        dropout_p_mpnn=0.0,
        model_lr_mpnn=1e-2,
        wd_mpnn=0.0,
        n_epochs=1,
        early_stopping=1,
        d_out=5,
        device='cpu',
        num_splits=1,
        log_training=False,
        dataset_name='cora',
    )
    assert 'val_acc_mean' in stats and 'test_acc_mean' in stats
    assert isinstance(results_list, list) and len(results_list) >= 1


def test_run_iterative_bridge_pipeline_smoke(tiny_graph):
    g = tiny_graph(n=6, undirected=True, k=3)
    P_k = np.eye(3)
    out = run_iterative_bridge_pipeline(
        g,
        P_k=P_k,
        model_type='GCN',
        h_feats_mpnn=8,
        n_layers_mpnn=1,
        dropout_p_mpnn=0.0,
        model_lr_mpnn=1e-2,
        wd_mpnn=0.0,
        n_epochs=1,
        early_stopping=1,
        d_out=5,
        device='cpu',
        seed=0,
        log_training=False,
        dataset_name='cora',
        do_self_loop=False,
        n_rewire=1,
        rewiring_method='bridge',
    )
    assert 'rewiring_history' in out
    assert len(out['rewiring_history']) == 1


def test_run_iterative_bridge_experiment_smoke(tiny_graph):
    g = tiny_graph(n=6, undirected=True, k=3)
    P_k = np.eye(3)
    stats, results = run_iterative_bridge_experiment(
        g,
        P_k=P_k,
        model_type='GCN',
        h_feats_mpnn=8,
        n_layers_mpnn=1,
        dropout_p_mpnn=0.0,
        model_lr_mpnn=1e-2,
        wd_mpnn=0.0,
        n_epochs=1,
        early_stopping=1,
        d_out=5,
        device='cpu',
        num_repeats=1,
        log_training=False,
        dataset_name='cora',
        do_self_loop=False,
    )
    assert 'val_acc_mean' in stats and 'test_acc_mean' in stats
    assert isinstance(results, list) and len(results) >= 1
