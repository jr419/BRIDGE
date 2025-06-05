"""
BRIDGE: Block Rewiring from Inference-Derived Graph Ensembles

A library for optimizing graph neural networks through inference-derived graph rewiring.
"""

from .models import GCN, HPGraphConv, SelectiveGCN
from .rewiring import run_bridge_pipeline, run_bridge_experiment
# ...

__version__ = "0.1.0"
