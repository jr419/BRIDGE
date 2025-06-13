# bridge/sensitivity/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats # Renamed to avoid conflict

def plot_local_sensitivity_validation(node_df, save_path, h_value_to_plot=0.5):
    """
    Plots node-level sensitivity condition satisfaction vs. actual MPNN>FNN accuracy improvement.
    Similar to Figure 2a in the paper. [cite: 115, 23]
    Requires a specific homophily value (h_value_to_plot) to select data for a single graph instance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter data for a specific homophily and run to get one graph instance
    # Taking the first run for the specified h_value
    plot_data = node_df[(node_df['homophily_h'] == h_value_to_plot) & (node_df['run_idx'] == 0)].copy()
    if plot_data.empty:
        print(f"No data found for h={h_value_to_plot} and run_idx=0 to generate plot 1.")
        plt.close(fig)
        return

    plot_data['gcn_fnn_acc_diff'] = plot_data['gcn_accuracy_node'] - plot_data['fnn_accuracy_node']

    # Subplot 1: Sensitivity Condition Satisfaction
    # Color nodes red/green for condition satisfied/not satisfied
    # This plot would typically need graph coordinates (e.g., from DGL graph object) to plot nodes.
    # For simplicity, we'll show a scatter plot of a node property colored by condition.
    # True graph plot would require passing graph object and using nx.draw or similar.
    colors_condition = ['green' if met else 'red' for met in plot_data['gcn_sensitivity_condition_met']]
    axes[0].scatter(plot_data['node_idx'], plot_data['degree'], c=colors_condition, alpha=0.6, s=10)
    axes[0].set_xlabel("Node Index (Placeholder for Graph Layout)")
    axes[0].set_ylabel("Node Degree")
    axes[0].set_title(f"Sensitivity Condition (h={h_value_to_plot})\nGreen: Satisfied, Red: Not")

    # Subplot 2: Actual GCN > FNN Accuracy
    # Color nodes red for GCN better, blue for FNN better (matching paper's Fig 2a right)
    acc_diff_colors = ['red' if diff > 0 else 'blue' if diff < 0 else 'grey' for diff in plot_data['gcn_fnn_acc_diff']]
    scatter_acc = axes[1].scatter(plot_data['node_idx'], plot_data['degree'], c=acc_diff_colors, cmap='coolwarm', vmin=-1, vmax=1, alpha=0.6, s=10)
    axes[1].set_xlabel("Node Index (Placeholder for Graph Layout)")
    axes[1].set_ylabel("Node Degree")
    axes[1].set_title(f"GCN vs FNN Accuracy (h={h_value_to_plot})\nRed: GCN better, Blue: FNN better")
    
    fig.colorbar(scatter_acc, ax=axes[1], label="Accuracy Diff (GCN - FNN)")
    fig.suptitle(f"Local Sensitivity Validation (h={h_value_to_plot})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot 1 saved to {save_path}")


def plot_snr_ratio_analysis(node_df, save_path):
    """
    Plots node-level SNR ratio (MPNN/FNN) vs. sensitivity condition satisfaction.
    Uses boxplots or violin plots to show distributions and t-test for significance.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    node_df['gcn_snr_mc_node_eff'] = node_df['gcn_snr_mc_node'].fillna(0)
    node_df['fnn_snr_mc_node_eff'] = node_df['fnn_snr_mc_node'].replace(0, 1e-9).fillna(1e-9) # Avoid div by zero

    node_df['snr_ratio_mc'] = node_df['gcn_snr_mc_node_eff'] / node_df['fnn_snr_mc_node_eff']
    
    # Cap SNR ratio for visualization if there are extreme outliers
    node_df['snr_ratio_mc_capped'] = np.clip(node_df['snr_ratio_mc'], -10, 10) # Example cap

    sns.boxplot(x='gcn_sensitivity_condition_met', y='snr_ratio_mc_capped', data=node_df, ax=ax)
    
    group_satisfied = node_df[node_df['gcn_sensitivity_condition_met'] == True]['snr_ratio_mc']
    group_not_satisfied = node_df[node_df['gcn_sensitivity_condition_met'] == False]['snr_ratio_mc']
    
    if len(group_satisfied) > 1 and len(group_not_satisfied) > 1:
        ttest_res = scipy_stats.ttest_ind(group_satisfied.dropna(), group_not_satisfied.dropna(), equal_var=False)
        ax.set_title(f"SNR Ratio (GCN/FNN) by Sensitivity Condition\nT-test p-value: {ttest_res.pvalue:.2e}")
    else:
        ax.set_title(f"SNR Ratio (GCN/FNN) by Sensitivity Condition\n(Not enough data for t-test)")
        
    ax.set_xlabel("Sensitivity Condition Satisfied")
    ax.set_ylabel("SNR Ratio (GCN_MC / FNN_MC) (Capped at +/-10)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot 2 saved to {save_path}")
    

def plot_node_acc_analysis(node_df, save_path):
    """
    Plots node-level SNR ratio (MPNN/FNN) vs. sensitivity condition satisfaction.
    Uses boxplots or violin plots to show distributions and t-test for significance.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    node_df['gcn_accuracy_node_eff'] = node_df['gcn_accuracy_node'].fillna(0)
    node_df['fnn_accuracy_node_eff'] = node_df['fnn_accuracy_node'].replace(0, 1e-9).fillna(1e-9) # Avoid div by zero

    node_df['node_accuracy_improvement'] = (node_df['gcn_accuracy_node_eff'] > node_df['fnn_accuracy_node_eff']).astype(int)
    
    group_satisfied = node_df[node_df['gcn_sensitivity_condition_met'] == True]['node_accuracy_improvement']
    group_not_satisfied = node_df[node_df['gcn_sensitivity_condition_met'] == False]['node_accuracy_improvement']
    
    #group by "node_idx"and average the accuracy improvement
    node_df['node_idx_avg'] = node_df.groupby('node_idx')['node_accuracy_improvement'].transform('mean')

    node_df_reduced = node_df[['gcn_sensitivity_condition_met', 'node_idx_avg', 'node_idx', 'homophily_h']].drop_duplicates()
    node_df_reduced = node_df_reduced.dropna(subset=['node_idx_avg'])
    node_df_reduced['gcn_sensitivity_condition_met'] = node_df_reduced['gcn_sensitivity_condition_met'].astype(str)
    node_df_reduced['node_idx_avg'] = node_df_reduced['node_idx_avg'].astype(float)

    sns.boxplot(x='gcn_sensitivity_condition_met', y='node_idx_avg', data=node_df_reduced, ax=ax)
    
    group_satisfied = node_df_reduced[node_df['gcn_sensitivity_condition_met'] == True]['node_idx_avg']
    group_not_satisfied = node_df_reduced[node_df['gcn_sensitivity_condition_met'] == False]['node_idx_avg']
    
    print(node_df_reduced)
    
    
    if len(group_satisfied) > 1 and len(group_not_satisfied) > 1:
        ttest_res = scipy_stats.ttest_ind(group_satisfied.dropna(), group_not_satisfied.dropna(), equal_var=False)
        ax.set_title(f"Accuracy improvement (GCN/FNN) by Sensitivity Condition\nT-test p-value: {ttest_res.pvalue:.2e}")
    else:
        ax.set_title(f"Accuracy improvement (GCN/FNN) by Sensitivity Condition\n(Not enough data for t-test)")
        
    ax.set_xlabel("Sensitivity Condition Satisfied")
    ax.set_ylabel("Accuracy improvement (GCN/FNN)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot 2 saved to {save_path}")

def plot_bottlenecking_snr_scatter(node_df, save_path):
    """
    Scatter plot of local within-class bottlenecking score vs. node-level SNR, colored by accuracy.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Average GCN accuracy per node across runs and h values for coloring
    # Or, pick a specific h for a clearer plot
    avg_node_acc = node_df.groupby('node_idx')['gcn_accuracy_node'].mean().reset_index()
    plot_data = pd.merge(node_df, avg_node_acc, on='node_idx', suffixes=('', '_avg'))

    # For a single representative plot, filter by a specific h and run_idx
    # plot_data_sample = plot_data[(plot_data['homophily_h'] == 0.5) & (plot_data['run_idx'] == 0)]
    # if plot_data_sample.empty:
    #     print("Not enough data for bottleneck-SNR scatter for h=0.5, run 0")
    #     plt.close(fig)
    #     return
    plot_data_sample = plot_data # Use all data points for now

    scatter = ax.scatter(
        plot_data_sample['gcn_bottleneck_score_node'], 
        plot_data_sample['gcn_snr_mc_node'], 
        c=plot_data_sample['gcn_accuracy_node_avg'], 
        cmap='viridis', alpha=0.5
    )
    
    # Trend line (optional, can be noisy)
    # m, b = np.polyfit(plot_data_sample['gcn_bottleneck_score_node'].dropna(), plot_data_sample['gcn_snr_mc_node'].dropna(), 1)
    # ax.plot(plot_data_sample['gcn_bottleneck_score_node'], m * plot_data_sample['gcn_bottleneck_score_node'] + b, color='red', linestyle='--')

    corr, p_val = scipy_stats.pearsonr(plot_data_sample['gcn_bottleneck_score_node'].dropna(), plot_data_sample['gcn_snr_mc_node'].dropna())
    
    ax.set_xlabel("Local Within-Class Bottlenecking Score (h_i^{l,l})")
    ax.set_ylabel("Node-level GCN SNR (MC)")
    ax.set_title(f"Bottlenecking Score vs. SNR (Colored by Avg Node Accuracy)\nPearson r: {corr:.2f}, p-value: {p_val:.2e}")
    fig.colorbar(scatter, label='Average GCN Node Accuracy')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot 3 saved to {save_path}")


def plot_graph_wide_snr_validation(graph_df, save_path, fnn_acc_mean=None, fnn_acc_std=None):
    """
    Plots graph-wide average SNR (Theorem 1 vs Monte Carlo) vs. edge homophily.
    Similar to Figure 2b in the paper. [cite: 115, 23]
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Group by homophily_h and calculate mean and std for SNR and accuracy
    summary_df = graph_df.groupby('homophily_h').agg(
        snr_mc_mean=('gcn_avg_snr_mc', 'mean'),
        snr_mc_std=('gcn_avg_snr_mc', 'std'),
        snr_theorem_mean=('gcn_avg_snr_theorem', 'mean'),
        snr_theorem_std=('gcn_avg_snr_theorem', 'std'),
        gcn_acc_mean=('gcn_test_accuracy_graph', 'mean'),
        gcn_acc_std=('gcn_test_accuracy_graph', 'std')
    ).reset_index()

    # Ensure stds are not NaN (e.g., if only one run_idx per h)
    summary_df = summary_df.fillna(0)

    color_mc = 'tab:blue'
    color_theorem = 'tab:orange'
    color_acc = 'tab:green'

    # Left y-axis: SNR
    ax1.set_xlabel('Edge Homophily (h)')
    ax1.set_ylabel('SNR Estimate (GCN)', color=color_mc)
    
    ln1 = ax1.plot(summary_df['homophily_h'], summary_df['snr_mc_mean'], label='Monte Carlo SNR (GCN)', color=color_mc, marker='o', linestyle='-')
    ax1.fill_between(summary_df['homophily_h'], summary_df['snr_mc_mean'] - summary_df['snr_mc_std'], summary_df['snr_mc_mean'] + summary_df['snr_mc_std'], color=color_mc, alpha=0.2)
    
    ln2 = ax1.plot(summary_df['homophily_h'], summary_df['snr_theorem_mean'], label='Sensitivity-based SNR (GCN)', color=color_theorem, marker='s', linestyle='-')
    ax1.fill_between(summary_df['homophily_h'], summary_df['snr_theorem_mean'] - summary_df['snr_theorem_std'], summary_df['snr_theorem_mean'] + summary_df['snr_theorem_std'], color=color_theorem, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color_mc)

    # Right y-axis: Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color=color_acc)
    ln3 = ax2.plot(summary_df['homophily_h'], summary_df['gcn_acc_mean'], label='GCN Test Accuracy', color=color_acc, linestyle='-', marker='d')
    ax2.fill_between(summary_df['homophily_h'], summary_df['gcn_acc_mean'] - summary_df['gcn_acc_std'], summary_df['gcn_acc_mean'] + summary_df['gcn_acc_std'], color=color_acc, alpha=0.15)
    
    # FNN Accuracy Baseline (from paper's Figure 2b, assuming it's relatively constant)
    # This should ideally be computed from the FNN runs in graph_df
    fnn_acc_data = graph_df.groupby('homophily_h')['fnn_test_accuracy_graph'].agg(['mean', 'std']).reset_index()
    if not fnn_acc_data.empty:
         # Plot FNN accuracy as a line if it varies with h, or a single horizontal line if constant
        ln4 = ax2.plot(fnn_acc_data['homophily_h'], fnn_acc_data['mean'], color='grey', linestyle='--', linewidth=1, label='FNN Test Accuracy')
        ax2.fill_between(fnn_acc_data['homophily_h'], fnn_acc_data['mean'] - fnn_acc_data['std'], fnn_acc_data['mean'] + fnn_acc_data['std'], color='grey', alpha=0.1)
    else: # Fallback if FNN data isn't available per h
        fnn_acc_mean_overall = graph_df['fnn_test_accuracy_graph'].mean()
        fnn_acc_std_overall = graph_df['fnn_test_accuracy_graph'].std()
        if pd.notna(fnn_acc_mean_overall):
            ln4 = ax2.axhline(y=fnn_acc_mean_overall, color='grey', linestyle='--', linewidth=1, label=f'FNN Test Acc (Avg: {fnn_acc_mean_overall:.2f})')
            ax2.fill_between(summary_df['homophily_h'], fnn_acc_mean_overall - fnn_acc_std_overall, fnn_acc_mean_overall + fnn_acc_std_overall, color='grey', alpha=0.1)


    ax2.tick_params(axis='y', labelcolor=color_acc)
    
    # Combine legends
    lns = ln1 + ln2 + ln3
    if 'ln4' in locals(): lns += ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Graph-wide SNR and Test Accuracy vs Edge Homophily")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot 4 saved to {save_path}")


def plot_snr_accuracy_correlation(graph_df, save_path):
    """
    Scatter plot of graph-wide average SNR vs. test accuracy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # SNR (MC) vs Accuracy
    sns.regplot(x='gcn_avg_snr_mc', y='gcn_test_accuracy_graph', data=graph_df, ax=axes[0], scatter_kws={'alpha':0.3})
    corr_mc, p_mc = scipy_stats.pearsonr(graph_df['gcn_avg_snr_mc'].dropna(), graph_df['gcn_test_accuracy_graph'].dropna())
    axes[0].set_xlabel("Graph-wide Average GCN SNR (Monte Carlo)")
    axes[0].set_ylabel("GCN Test Accuracy")
    axes[0].set_title(f"MC SNR vs. Accuracy\nPearson r: {corr_mc:.2f}, p: {p_mc:.2e}")

    # SNR (Theorem) vs Accuracy
    sns.regplot(x='gcn_avg_snr_theorem', y='gcn_test_accuracy_graph', data=graph_df, ax=axes[1], scatter_kws={'alpha':0.3})
    corr_th, p_th = scipy_stats.pearsonr(graph_df['gcn_avg_snr_theorem'].dropna(), graph_df['gcn_test_accuracy_graph'].dropna())
    axes[1].set_xlabel("Graph-wide Average GCN SNR (Theorem)")
    axes[1].set_ylabel("") # Shared Y-axis
    axes[1].set_title(f"Theorem SNR vs. Accuracy\nPearson r: {corr_th:.2f}, p: {p_th:.2e}")
    
    fig.suptitle("Correlation between Graph-wide SNR and GCN Test Accuracy", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot 5 saved to {save_path}")