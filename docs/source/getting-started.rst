Getting Started
===============

Installation
-----------

You can install BRIDGE directly from the repository:

.. code-block:: bash

   git clone https://github.com/jr419/BRIDGE.git
   cd BRIDGE
   pip install -e .

This will install BRIDGE in development mode, making it available in your Python environment.

Requirements
~~~~~~~~~~~

BRIDGE requires the following Python packages:

.. code-block:: text

   dgl>=1.1.0
   numpy>=1.26.2
   optuna>=4.0.0
   ortools>=9.11.4210
   pandas>=2.1.3
   scipy>=1.11.3
   scikit-learn>=1.3.2
   torch>=2.0.0
   tqdm>=4.66.1
   pyyaml>=5.4.1

These dependencies will be automatically installed when using ``pip install -e .``.

Quick Start Example
------------------

Here's a simple example to get started with the BRIDGE rewiring technique:

.. code-block:: python

   import dgl
   import torch
   from bridge.models import GCN
   from bridge.rewiring import run_bridge_pipeline
   from bridge.utils import generate_all_symmetric_permutation_matrices

   # Load a dataset
   dataset = dgl.data.CoraGraphDataset()
   g = dataset[0]

   # Generate permutation matrices
   k = len(torch.unique(g.ndata['label']))
   all_matrices = generate_all_symmetric_permutation_matrices(k)
   P_k = all_matrices[0]  # Choose the first permutation matrix

   # Run the rewiring pipeline
   results = run_bridge_pipeline(
       g=g,
       P_k=P_k,
       h_feats_gcn=64,
       n_layers_gcn=2,
       dropout_p_gcn=0.5,
       model_lr_gcn=1e-3,
       h_feats_selective=64,
       n_layers_selective=2,
       dropout_p_selective=0.5,
       model_lr_selective=1e-3,
       num_graphs=1,
       device='cuda' if torch.cuda.is_available() else 'cpu'
   )

   # Print results
   print(f"Base GCN accuracy: {results['cold_start']['test_acc']:.4f}")
   print(f"Selective GCN accuracy: {results['selective']['test_acc']:.4f}")

Sensitivity Analysis Example
---------------------------

To analyze the Signal-to-Noise Ratio (SNR) and sensitivity of a graph neural network:

.. code-block:: python

   import torch
   import dgl
   from bridge.sensitivity import (
       estimate_snr_theorem,
       estimate_sensitivity_autograd,
       run_sensitivity_experiment,
       plot_snr_vs_homophily
   )

   # Load a dataset
   dataset = dgl.data.CoraGraphDataset()
   g = dataset[0]

   # Configure sensitivity analysis
   feature_params = {
       'intra_class_cov': 0.1,
       'inter_class_cov': -0.05,
       'global_cov': 1.0,
       'noise_cov': 1.0,
       'feature_dim': 5
   }

   # Run experiment across multiple graphs with varying homophily
   results = run_sensitivity_experiment(
       g, 
       homophily_values=[0.1, 0.3, 0.5, 0.7, 0.9],
       feature_params=feature_params
   )

   # Visualize results
   plot_snr_vs_homophily(results)

Command-Line Interface
---------------------

BRIDGE provides a command-line interface for running experiments:

.. code-block:: bash

   # Run a rewiring experiment on a standard dataset
   python -m bridge.main --dataset_type standard --standard_datasets cora --num_trials 100 --experiment_name cora_experiment

   # Run a rewiring experiment on a synthetic dataset
   python -m bridge.main --dataset_type synthetic --syn_homophily 0.3 --syn_nodes 3000 --syn_classes 4 --experiment_name synthetic_experiment

   # Run a sensitivity analysis experiment with a configuration file
   python -m bridge.main --experiment_type sensitivity --config config_examples/snr_analysis.yaml

See the :doc:`CLI Reference <cli-reference>` for more options.
