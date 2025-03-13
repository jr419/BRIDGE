Theoretical Framework
===================

Signal-to-Noise Ratio Framework
------------------------------

The core of our theoretical framework is the Signal-to-Noise Ratio (SNR) analysis of Message Passing Neural Networks (MPNNs). This approach allows us to quantify how effectively an MPNN can separate different node classes based on the underlying graph structure.

Feature Decomposition
~~~~~~~~~~~~~~~~~~~~

We model the distribution of node features in relation to their class labels using a decomposition into signal and noise components:

.. math::

   X_j = \mu_{y_j} + \gamma + \epsilon_j

where:

- :math:`\mu_{y_j}` represents the class-specific signal for class :math:`y_j`
- :math:`\gamma` is a global shift vector shared across all nodes (with zero mean)
- :math:`\epsilon_j` is node-specific IID noise (with zero mean)

Signal-to-Noise Ratio
~~~~~~~~~~~~~~~~~~~~

For an :math:`\ell`-layer MPNN, we define the SNR as:

.. math::

   \text{SNR}(H^{(\ell)}_{ip}) := \frac{\text{Var}_\mu[\mathbb{E}[H^{(\ell)}_{ip} | \mu]]}{\mathbb{E}_\mu[\text{Var}[H^{(\ell)}_{ip} | \mu]]}

This measures the ratio of variance explained by class-specific signals to the unexplained variance, quantifying how well the model separates different classes.

Sensitivity Measures
~~~~~~~~~~~~~~~~~~~

We introduce three key metrics to analyze MPNN behavior:

1. **Signal Sensitivity**: Measures the model's response to coherent class-specific changes in the input features

   .. math::

      S^{(\ell)}_{i,p,q,r} := \sum_{j,k \in V} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{jq}} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{kr}} \delta_{y_j,y_k}

2. **Noise Sensitivity**: Measures the model's response to unstructured, IID noise

   .. math::

      N^{(\ell)}_{i,p,q,r} := \sum_{j \in V} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{jq}} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{jr}}

3. **Global Sensitivity**: Measures the model's response to global shifts in the input

   .. math::

      T^{(\ell)}_{i,p,q,r} := \sum_{j,k \in V} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{jq}} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{kr}}

SNR-Sensitivity Relation
~~~~~~~~~~~~~~~~~~~~~~~

Our main theoretical result relates the SNR to the sensitivity measures:

.. math::

   \text{SNR}(H^{(\ell)}_{ip}) \approx \frac{\sum_{q,r=1}^{d_{out}}(\Sigma^{(intra)}_{qr} - \Sigma^{(inter)}_{qr})S^{(\ell)}_{i,p,q,r} + \sum_{q,r=1}^{d_{out}}\Sigma^{(inter)}_{qr}T^{(\ell)}_{i,p,q,r}}{\sum_{q,r=1}^{d_{out}}\Phi_{qr}T^{(\ell)}_{i,p,q,r} + \sum_{q,r=1}^{d_{out}}\Psi_{qr}N^{(\ell)}_{i,p,q,r}}

where :math:`\Sigma^{(intra)}`, :math:`\Sigma^{(inter)}`, :math:`\Phi`, and :math:`\Psi` are the covariance matrices of intra-class signals, inter-class signals, global shifts, and node noise, respectively.

Higher-Order Homophily
---------------------

Definition
~~~~~~~~~

We introduce higher-order homophily as a generalization of standard homophily measures. For a graph shift operator :math:`\hat{A}`, the :math:`r`-order homophily is defined as:

.. math::

   h_r(\hat{A}) := \frac{1}{n}\sum_{i,j \in V} [\hat{A}^r]_{ij} \delta_{y_i,y_j}

Similarly, we define :math:`r`-order self-connectivity :math:`\eta_r(\hat{A})` and total connectivity :math:`\tau_r(\hat{A})`.

Sensitivity Bounds
~~~~~~~~~~~~~~~~

Our analysis shows that the signal sensitivity of an MPNN is bounded by higher-order homophily. For an isotropic MPNN with bounded message and update functions, we prove:

.. math::

   S^{(\ell)}_{i,p,q,r} \leq \sum_{s,t=0}^{\ell} \binom{\ell}{s}\binom{\ell}{t} \alpha_1^{2\ell-s-t}(\alpha_2\beta)^{s+t} h^{s,t}_i(\hat{A})

where :math:`h^{s,t}_i(\hat{A})` is the local :math:`(s,t)`-order homophily at node :math:`i`. This bound reveals that higher-order homophily fundamentally limits the ability of MPNNs to leverage class-specific signals.

Homophilic Bottlenecks
---------------------

Definition
~~~~~~~~~

Homophilic bottlenecks are nodes with low local homophily, which restrict the flow of class-specific information through the network. Unlike general bottlenecks, which impede all information flow, homophilic bottlenecks specifically impact nodes from the same class.

Underreaching and Oversquashing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We decompose the effects of homophilic bottlenecks into two distinct phenomena:

1. **Underreaching**: The failure of information to propagate between distant nodes of the same class
2. **Oversquashing**: The interference and compression of information from multiple source nodes as it converges at a target node

We formalize this decomposition for sparse graph ensembles as:

.. math::

   \mathbb{E}[\hat{A}^{\ell}_{ij}] = \sum_{r=1}^{\ell} \mathbb{E}[\hat{A}^{\ell}_{ij} | \lambda_{ij} = r] \cdot P(\lambda_{ij} = r)

where :math:`\lambda_{ij}` is the shortest path length between nodes :math:`i` and :math:`j`.

Optimal Graph Structures
-----------------------

Stochastic Block Model
~~~~~~~~~~~~~~~~~~~~~

For graph ensembles following a Stochastic Block Model (SBM), we derive the optimal block matrix :math:`B` that maximizes higher-order homophily:

.. math::

   B = \frac{\langle d \rangle}{k} \Pi^{-1} P_k \Pi^{-1}

where :math:`\Pi` is the diagonal matrix of class proportions, :math:`\langle d \rangle` is the desired mean degree, and :math:`P_k` is a symmetric permutation matrix.

Structure Characterization
~~~~~~~~~~~~~~~~~~~~~~~~

The optimal graph structures for MPNNs are precisely those formed by unions of fully connected single-class clusters and bipartite connections between different classes. This result guides our graph rewiring strategy to enhance MPNN performance.
