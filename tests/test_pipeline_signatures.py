from inspect import signature
import pytest

try:
    from bridge.rewiring.pipeline import (
        run_bridge_pipeline,
        run_bridge_experiment,
        run_iterative_bridge_pipeline,
        run_iterative_bridge_experiment,
    )
except Exception as e:
    pytest.skip(f"Skipping pipeline signature tests due to DGL import failure: {e}", allow_module_level=True)


def test_signatures_no_temperature_or_padd_remove():
    for fn in [run_bridge_pipeline, run_bridge_experiment, run_iterative_bridge_pipeline, run_iterative_bridge_experiment]:
        sig = signature(fn)
        params = list(sig.parameters.keys())
        joined = ','.join(params)
        assert 'temperature' not in joined
        assert 'p_add' not in joined
        assert 'p_remove' not in joined
