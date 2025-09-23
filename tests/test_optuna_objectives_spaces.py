import numpy as np
import dgl
import torch

from bridge.optimization.optuna_objectives import objective_rewiring, objective_iterative_rewiring


def tiny_graph():
    g = dgl.graph(([0,1,2],[1,2,0]), num_nodes=3)
    g = dgl.to_bidirected(g)
    g.ndata['feat'] = torch.eye(3)
    g.ndata['label'] = torch.tensor([0,1,2])
    g.ndata['train_mask'] = torch.tensor([True, True, True])
    g.ndata['val_mask'] = torch.tensor([True, True, True])
    g.ndata['test_mask'] = torch.tensor([True, True, True])
    return g


def test_objective_callable_smoke(monkeypatch):
    # Create a dummy Trial with minimal API for suggest methods
    class DummyTrial:
        def __init__(self):
            self.params = {}
            self.user_attrs = {}
        def suggest_int(self, name, low, high):
            self.params[name] = low
            return low
        def suggest_float(self, name, low, high, log=False):
            self.params[name] = low
            return low
        def suggest_categorical(self, name, choices):
            val = choices[0]
            self.params[name] = val
            return val
        def set_user_attr(self, name, value):
            self.user_attrs[name] = value

    g = tiny_graph()
    all_mats = [np.eye(3)]
    trial = DummyTrial()

    # Just ensure it runs end-to-end without referencing removed params
    try:
        objective_rewiring(
            trial, g,
            best_mpnn_params={'h_feats': 8, 'n_layers': 1, 'dropout_p': 0.0, 'model_lr': 1e-2, 'weight_decay': 0.0},
            all_matrices=all_mats,
            device='cpu',
            n_epochs=1,
            num_splits=1,
            early_stopping=1,
            do_hp=False,
            do_self_loop=False,
            dataset_name='cora'
        )
    except Exception as e:
        assert False, f"objective_rewiring raised: {e}"

    trial2 = DummyTrial()
    try:
        objective_iterative_rewiring(
            trial2, g,
            best_mpnn_params={'h_feats': 8, 'n_layers': 1, 'dropout_p': 0.0, 'model_lr': 1e-2, 'weight_decay': 0.0},
            all_matrices=all_mats,
            device='cpu',
            n_epochs=1,
            num_splits=1,
            early_stopping=1,
            do_hp=False,
            do_self_loop=False,
            dataset_name='cora',
            rewiring_method='bridge',
            n_rewire_iterations_range=[1, 1]
        )
    except Exception as e:
        assert False, f"objective_iterative_rewiring raised: {e}"
