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
    available_h = node_df['homophily_h'].dropna().unique()
    if len(available_h) == 0:
        print("No homophily values found to generate plot 1.")
        return
    h_to_plot = available_h[np.argmin(np.abs(available_h - h_value_to_plot))]
    plot_data = node_df[(node_df['homophily_h'] == h_to_plot) & (node_df['run_idx'] == 0)].copy()
    if plot_data.empty:
        print(f"No data found for h={h_to_plot} and run_idx=0 to generate plot 1.")
        return

    backbones = sorted(plot_data['backbone'].dropna().unique()) if 'backbone' in plot_data.columns else [None]
    fig, axes = plt.subplots(len(backbones), 2, figsize=(12, 4 * len(backbones)), squeeze=False)

    for row_idx, backbone in enumerate(backbones):
        if backbone is None:
            backbone_data = plot_data.copy()
            title_prefix = ""
        else:
            backbone_data = plot_data[plot_data['backbone'] == backbone].copy()
            title_prefix = f"{backbone}\n"
        backbone_data['gcn_fnn_acc_diff'] = (
            backbone_data['gcn_accuracy_node'].astype(int)
            - backbone_data['fnn_accuracy_node'].astype(int)
        )

        colors_condition = ['green' if met else 'red' for met in backbone_data['gcn_sensitivity_condition_met']]
        axes[row_idx, 0].scatter(backbone_data['node_idx'], backbone_data['degree'], c=colors_condition, alpha=0.6, s=10)
        axes[row_idx, 0].set_xlabel("Node Index")
        axes[row_idx, 0].set_ylabel("Node Degree")
        axes[row_idx, 0].set_title(f"{title_prefix}Sensitivity Condition (h={h_to_plot:.3f})")

        acc_diff_colors = ['red' if diff > 0 else 'blue' if diff < 0 else 'grey' for diff in backbone_data['gcn_fnn_acc_diff']]
        axes[row_idx, 1].scatter(backbone_data['node_idx'], backbone_data['degree'], c=acc_diff_colors, alpha=0.6, s=10)
        axes[row_idx, 1].set_xlabel("Node Index")
        axes[row_idx, 1].set_ylabel("Node Degree")
        axes[row_idx, 1].set_title(f"{title_prefix}GCN vs FNN Accuracy")

    fig.suptitle(f"Local Sensitivity Validation (h={h_to_plot:.3f})", fontsize=16)
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

    hue = 'backbone' if 'backbone' in node_df.columns else None
    sns.boxplot(x='gcn_sensitivity_condition_met', y='snr_ratio_mc_capped', hue=hue, data=node_df, ax=ax)
    
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
    group_cols = ['node_idx', 'homophily_h']
    if 'backbone' in node_df.columns:
        group_cols.insert(0, 'backbone')
    node_df['node_idx_avg'] = node_df.groupby(group_cols)['node_accuracy_improvement'].transform('mean')

    keep_cols = ['gcn_sensitivity_condition_met', 'node_idx_avg', 'node_idx', 'homophily_h']
    if 'backbone' in node_df.columns:
        keep_cols.insert(0, 'backbone')
    node_df_reduced = node_df[keep_cols].drop_duplicates()
    node_df_reduced = node_df_reduced.dropna(subset=['node_idx_avg'])
    node_df_reduced['gcn_sensitivity_condition_met'] = node_df_reduced['gcn_sensitivity_condition_met'].astype(str)
    node_df_reduced['node_idx_avg'] = node_df_reduced['node_idx_avg'].astype(float)

    hue = 'backbone' if 'backbone' in node_df_reduced.columns else None
    sns.boxplot(x='gcn_sensitivity_condition_met', y='node_idx_avg', hue=hue, data=node_df_reduced, ax=ax)
    
    group_satisfied = node_df_reduced[node_df_reduced['gcn_sensitivity_condition_met'] == 'True']['node_idx_avg']
    group_not_satisfied = node_df_reduced[node_df_reduced['gcn_sensitivity_condition_met'] == 'False']['node_idx_avg']
    
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
    backbones = sorted(node_df['backbone'].dropna().unique()) if 'backbone' in node_df.columns else [None]
    fig, axes = plt.subplots(1, len(backbones), figsize=(8 * len(backbones), 6), squeeze=False)
    
    # Average GCN accuracy per node across runs and h values for coloring
    # Or, pick a specific h for a clearer plot
    avg_cols = ['node_idx']
    if 'backbone' in node_df.columns:
        avg_cols.insert(0, 'backbone')
    avg_node_acc = node_df.groupby(avg_cols)['gcn_accuracy_node'].mean().reset_index()
    plot_data = pd.merge(node_df, avg_node_acc, on=avg_cols, suffixes=('', '_avg'))

    for idx, backbone in enumerate(backbones):
        ax = axes[0, idx]
        plot_data_sample = plot_data if backbone is None else plot_data[plot_data['backbone'] == backbone]
        scatter = ax.scatter(
            plot_data_sample['gcn_bottleneck_score_node'],
            plot_data_sample['gcn_snr_mc_node'],
            c=plot_data_sample['gcn_accuracy_node_avg'],
            cmap='viridis', alpha=0.5
        )
        valid_data = plot_data_sample[['gcn_bottleneck_score_node', 'gcn_snr_mc_node']].dropna()
        x = valid_data['gcn_bottleneck_score_node']
        y = valid_data['gcn_snr_mc_node']
        if len(x) > 1 and len(y) > 1 and x.nunique() > 1 and y.nunique() > 1:
            corr, p_val = scipy_stats.pearsonr(x, y)
            stats_text = f"Pearson r: {corr:.2f}, p-value: {p_val:.2e}"
        else:
            stats_text = "Not enough variation for Pearson r"
        ax.set_xlabel("Local Within-Class Bottlenecking Score (h_i^{l,l})")
        ax.set_ylabel("Node-level GCN SNR (MC)")
        ax.set_title(f"{backbone or 'GCN'}\n{stats_text}")
        fig.colorbar(scatter, ax=ax, label='Average GCN Node Accuracy')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot 3 saved to {save_path}")


def plot_graph_wide_snr_validation(graph_df, save_path, fnn_acc_mean=None, fnn_acc_std=None):
    """
    Plots graph-wide average SNR (Theorem 1 vs Monte Carlo) vs. edge homophily.
    Similar to Figure 2b in the paper. [cite: 115, 23]
    """
    fig, ax1 = plt.subplots(figsize=(11, 7))
    
    # Group by homophily_h and calculate mean and std for SNR and accuracy
    group_cols = ['homophily_h']
    if 'backbone' in graph_df.columns:
        group_cols.insert(0, 'backbone')
    summary_df = graph_df.groupby(group_cols).agg(
        snr_mc_mean=('gcn_avg_snr_mc', 'mean'),
        snr_mc_std=('gcn_avg_snr_mc', 'std'),
        snr_theorem_mean=('gcn_avg_snr_theorem', 'mean'),
        snr_theorem_std=('gcn_avg_snr_theorem', 'std'),
        gcn_acc_mean=('gcn_test_accuracy_graph', 'mean'),
        gcn_acc_std=('gcn_test_accuracy_graph', 'std')
    ).reset_index().fillna(0)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Left y-axis: SNR
    ax1.set_xlabel('Edge Homophily (h)')
    ax1.set_ylabel('SNR Estimate (GCN)')

    backbones = sorted(summary_df['backbone'].dropna().unique()) if 'backbone' in summary_df.columns else [None]
    lines = []
    for idx, backbone in enumerate(backbones):
        data = summary_df if backbone is None else summary_df[summary_df['backbone'] == backbone]
        color = colors[idx % len(colors)]
        label_prefix = backbone or 'GCN'
        lines += ax1.plot(data['homophily_h'], data['snr_mc_mean'], label=f'{label_prefix} MC SNR', color=color, marker='o', linestyle='-')
        lines += ax1.plot(data['homophily_h'], data['snr_theorem_mean'], label=f'{label_prefix} theorem SNR', color=color, marker='s', linestyle='--')

    # Right y-axis: Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy')
    for idx, backbone in enumerate(backbones):
        data = summary_df if backbone is None else summary_df[summary_df['backbone'] == backbone]
        color = colors[idx % len(colors)]
        label_prefix = backbone or 'GCN'
        lines += ax2.plot(data['homophily_h'], data['gcn_acc_mean'], label=f'{label_prefix} test accuracy', color=color, linestyle=':', marker='d')
    
    # FNN Accuracy Baseline (from paper's Figure 2b, assuming it's relatively constant)
    # This should ideally be computed from the FNN runs in graph_df
    fnn_acc_data = graph_df.groupby('homophily_h')['fnn_test_accuracy_graph'].agg(['mean', 'std']).reset_index().fillna(0)
    if not fnn_acc_data.empty:
         # Plot FNN accuracy as a line if it varies with h, or a single horizontal line if constant
        lines += ax2.plot(fnn_acc_data['homophily_h'], fnn_acc_data['mean'], color='grey', linestyle='-.', linewidth=1, label='FNN Test Accuracy')
        ax2.fill_between(fnn_acc_data['homophily_h'], fnn_acc_data['mean'] - fnn_acc_data['std'], fnn_acc_data['mean'] + fnn_acc_data['std'], color='grey', alpha=0.1)
    else: # Fallback if FNN data isn't available per h
        fnn_acc_mean_overall = graph_df['fnn_test_accuracy_graph'].mean()
        fnn_acc_std_overall = graph_df['fnn_test_accuracy_graph'].std()
        if pd.notna(fnn_acc_mean_overall):
            lines.append(ax2.axhline(y=fnn_acc_mean_overall, color='grey', linestyle='-.', linewidth=1, label=f'FNN Test Acc (Avg: {fnn_acc_mean_overall:.2f})'))
            ax2.fill_between(summary_df['homophily_h'], fnn_acc_mean_overall - fnn_acc_std_overall, fnn_acc_mean_overall + fnn_acc_std_overall, color='grey', alpha=0.1)

    labs = [line.get_label() for line in lines]
    ax1.legend(lines, labs, loc='best', fontsize=9)
    
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
    hue = 'backbone' if 'backbone' in graph_df.columns else None

    # SNR (MC) vs Accuracy
    if hue:
        sns.scatterplot(x='gcn_avg_snr_mc', y='gcn_test_accuracy_graph', hue=hue, data=graph_df, ax=axes[0], alpha=0.5)
    else:
        sns.regplot(x='gcn_avg_snr_mc', y='gcn_test_accuracy_graph', data=graph_df, ax=axes[0], scatter_kws={'alpha':0.3})
    corr_mc, p_mc = scipy_stats.pearsonr(graph_df['gcn_avg_snr_mc'].dropna(), graph_df['gcn_test_accuracy_graph'].dropna())
    axes[0].set_xlabel("Graph-wide Average GCN SNR (Monte Carlo)")
    axes[0].set_ylabel("GCN Test Accuracy")
    axes[0].set_title(f"MC SNR vs. Accuracy\nPearson r: {corr_mc:.2f}, p: {p_mc:.2e}")

    # SNR (Theorem) vs Accuracy
    if hue:
        sns.scatterplot(x='gcn_avg_snr_theorem', y='gcn_test_accuracy_graph', hue=hue, data=graph_df, ax=axes[1], alpha=0.5, legend=False)
    else:
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
