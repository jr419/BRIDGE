from importlib import import_module


def test_rewiring_exports():
    mod = import_module('bridge.rewiring')
    exported = set(getattr(mod, '__all__', []))
    # Expected public API
    expected = {
        'create_rewired_graph',
        'run_bridge_pipeline',
        'run_bridge_experiment',
        'run_iterative_bridge_pipeline',
        'run_iterative_bridge_experiment',
        'sdrf_rewire',
        'digl_rewired',
    }
    assert expected.issubset(exported)


def test_models_exports():
    mod = import_module('bridge.models')
    exported = set(getattr(mod, '__all__', []))
    # HPGraphConv/SGC were removed; only GCN remains public
    assert {'GCN'}.issubset(exported)
