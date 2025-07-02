# import os
# import json
# from pathlib import Path
# from textwrap import indent

# # ------------------------------------------------------------------------------
# # Configuration
# # ------------------------------------------------------------------------------
# MAIN_DATASETS = ["cora", "citeseer", "pubmed"]
# OTHER_DATASETS = ['actor','squirrel','chameleon','wisconsin','cornell','texas']
# # OTHER_DATASETS = ['synthetic_graph_dataset_h=0.10_d=20.00',
# #                   'synthetic_graph_dataset_h=0.20_d=20.00',
# #                   'synthetic_graph_dataset_h=0.30_d=20.00',
# #                   'synthetic_graph_dataset_h=0.40_d=20.00',
# #                   'synthetic_graph_dataset_h=0.50_d=20.00',
# #                   'synthetic_graph_dataset_h=0.60_d=20.00',
# #                   'synthetic_graph_dataset_h=0.70_d=20.00'
# #                 ]
                  
# # If your dataset names have different cases or underscores,
# # adjust them here as needed. Also, if your folder structure differs,
# # you can change how parse_results finds and reads them.

# RESULTS_FILENAME = "results.json"

# # ------------------------------------------------------------------------------
# # Helper to parse a single dataset's JSON file
# # ------------------------------------------------------------------------------

# def parse_single_dataset_results(json_path):
#     """
#     Reads a single JSON results file and returns a dict with:
#       {
#         'base_acc': float,
#         'rewired_acc': float,
#         'improvement': float
#       }
#     using the test_accuracy (or test_accuracy_mean) from your JSON structure.
#     """
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     # For example, from your sample JSON, we can extract:
#     base_acc = 100*data["base_gcn"]["test_accuracy"]  # test_accuracy
#     base_acc_ci = data["base_gcn"]["test_accuracy_ci"]
#     # For "rewiring", if there's a mean and confidence interval, we can either
#     # use the mean or store the full ± CI. We'll store the mean plus the ± from CI.
#     rewired_mean = 100*data["rewiring"]["test_accuracy_mean"]
#     # If you have a test_accuracy_ci, we can format that as well:
#     rewired_acc_ci = data["rewiring"]["test_accuracy_ci"]  # [low, high]

#     # A simple way to format is "mean ± std", 
#     # but if you want to store the exact CI in LaTeX, you can do that too.
#     # Here, we’ll do something like "mean ± half_range":
#     base_half_range = 100*(base_acc_ci[1] - base_acc_ci[0]) / 2.0
#     rewired_half_range = 100*(rewired_acc_ci[1] - rewired_acc_ci[0]) / 2.0
#     rewired_str = f"{rewired_mean:.2f} ± {rewired_half_range:.2f}"
#     base_str = f"{base_acc:.2f} ± {base_half_range:.2f}"
    
#     # Or if you want the raw mean as a float:
#     rewired_acc = rewired_mean
    
#     # improvement percentage (already in your JSON as improvement_percentage?)
#     improvement = data["improvement_percentage"]
    
#     return {
#         "base_acc": base_acc,
#         "rewired_acc": rewired_acc,
#         "base_str": base_str,  # store the string for LaTeX
#         "rewired_str": rewired_str,  # store the string for LaTeX
#         "improvement": improvement,
#     }

# # ------------------------------------------------------------------------------
# # Parse results for an entire folder (i.e., one "model family")
# # ------------------------------------------------------------------------------

# def parse_results_for_model(folder_path):
#     """
#     Given a folder_path which contains subfolders or files for each dataset
#     (e.g. "cora/results.json", "citeseer/results.json", etc.),
#     parse all available datasets and return a dictionary:
#         {
#             'cora': {...},
#             'citeseer': {...},
#             'pubmed': {...},
#             'actor': {...},
#             ...
#         }
#     where each value is a dict with 'base_acc', 'rewired_acc', and 'improvement'.
#     """
#     results_dict = {}
#     folder_path = Path(folder_path)
    
#     # We assume each dataset is in a subfolder named exactly "cora", "citeseer", etc.
#     # If, instead, you have something like "cora.json" directly in folder_path,
#     # you can adapt the code to look for JSON files matching dataset names.
    
#     for json_file in folder_path.iterdir():
#         if json_file.exists() and os.path.basename(json_file).endswith('_results.json'):
#             dataset_name = os.path.splitext(os.path.basename(json_file))[0].replace('_results', '').replace('_sym', '').lower()  # e.g. "cora", "citeseer"
#             print(dataset_name)
#             results_dict[dataset_name] = parse_single_dataset_results(json_file)
    
#     return results_dict

# # ------------------------------------------------------------------------------
# # Build LaTeX table for main datasets (Cora, Citeseer, Pubmed)
# # ------------------------------------------------------------------------------
# def build_main_table(models_data, model_order=None, caption="", label=""):
#     """
#     models_data is a dict:
#        {
#          'ModelDisplayName1': {'cora': {...}, 'citeseer': {...}, ...},
#          'ModelDisplayName2': {...}
#        }
#     model_order is a list specifying the order of model rows in the table.
#     We produce a LaTeX table with columns: Model, Cora, Citeseer, Pubmed
#     and row groups: (Base, Rewired, Improvement) for each model.
    
#     We'll assume that each model_data dict has "cora", "citeseer", "pubmed" keys
#     with the standard sub-keys: 'base_acc', 'rewired_str', 'improvement', etc.
#     """
#     if model_order is None:
#         model_order = list(models_data.keys())
    
#     # You can adapt columns to your needs
#     table_header = r"""
# \begin{table}[htbp]
# \centering
# \caption{%s}
# \label{%s}
# \makebox[\textwidth]{\begin{tabular}{lccc}
# \toprule
# Model & Cora & Citeseer & Pubmed \\
# \midrule
# """ % (caption, label)
    
#     table_rows = []
    
#     for model_name in model_order:
#         # The dict for this model
#         md = models_data[model_name]
#         print(md)
#         # model_name = model_name.replace('High-Pass','Low-Pass')
        
#         # For each dataset in MAIN_DATASETS, fetch the data.
#         cora = md.get("cora", {})
#         citeseer = md.get("citeseer", {})
#         pubmed = md.get("pubmed", {})
        
#         # If a dataset is missing, you could handle that gracefully
#         # (e.g. put --)
#         def format_base_acc(x):
#             return x["base_str"] if x else "--"
        
#         def format_rewired_acc_str(x):
#             return x["rewired_str"] if x else "--"
        
#         def format_improvement(x):
#             return f"{x['improvement']:.2f}" if x else "--"
        
#         # row 1: Base
#         row_base = (
#             f"{model_name} (Base) & " 
#             f"{format_base_acc(cora)} & "
#             f"{format_base_acc(citeseer)} & "
#             f"{format_base_acc(pubmed)} \\\\"
#         )
        
#         # row 2: Rewired
#         row_rewired = (
#             f"{model_name} (Rewired) & "
#             f"{format_rewired_acc_str(cora)} & "
#             f"{format_rewired_acc_str(citeseer)} & "
#             f"{format_rewired_acc_str(pubmed)} \\\\"
#         )
        
#         # row 3: Improvement
#         row_improvement = (
#             f"{model_name} (Improvement \\%) & "
#             f"{format_improvement(cora)} & "
#             f"{format_improvement(citeseer)} & "
#             f"{format_improvement(pubmed)} \\\\"
#         )
        
#         table_rows.extend([row_base, row_rewired, row_improvement, r"\midrule"])
    
#     table_footer = r"""
# \bottomrule
# \end{tabular}}
# \end{table}
# """
    
#     table_content = table_header + "\n".join(table_rows) + table_footer
#     return table_content

# # ------------------------------------------------------------------------------
# # Build LaTeX table for "other" datasets (everything else)
# # ------------------------------------------------------------------------------
# def build_other_table(models_data, model_order=None, caption="", label=""):
#     """
#     Very similar to build_main_table, but we gather columns for everything
#     in each model’s dict that is *not* in MAIN_DATASETS.
#     Then produce one row per dataset, or produce a wide table with all columns.
    
#     In your sample, you show a table with columns for:
#       Actor, Chameleon, Squirrel, Texas, Wisconsin, Cornell
#     and rows for Base, Rewired, Improvement.
    
#     For demonstration, we’ll assume we know the "other" dataset list or
#     we can find them from the first model’s data keys.
#     """
#     if model_order is None:
#         model_order = list(models_data.keys())
    
#     # Collect all "other" dataset names from the first model
#     # (you can unify them across all models if needed).
#     all_datasets = set()
#     for model_name in model_order:
#         all_datasets.update(models_data[model_name].keys())
#     # Filter out main
#     # Sort them as you like:
#     # e.g. ds_order = ["actor", "chameleon", ...] if you want a fixed order:
#     ds_order = OTHER_DATASETS
    
#     # Build the header line for dataset columns:
#     header_columns = " & ".join(name.capitalize() for name in ds_order)
    
#     table_header = r"""
# \begin{table}[htbp]
# \centering
# \caption{%s}
# \label{%s}
# \makebox[\textwidth]{\begin{tabular}{l%s}
# \toprule
# Model & %s \\
# \midrule
# """ % (caption, label, "c" * len(ds_order), header_columns)
    
#     table_rows = []
    
#     for model_name in model_order:
#         md = models_data[model_name]
        
#         # We'll create three lines: Base, Rewired, Improvement
#         # across all "other" dataset columns
#         base_line = [f"{model_name} (Base)"]
#         rewired_line = [f"{model_name} (Rewired)"]
#         improv_line = [f"{model_name} (Improvement \\%)"]
        
#         for ds in ds_order:
            
#             entry = md.get(ds, {})
#             if entry:
#                 base_str = entry.get("base_str", None)
#                 rewired_str = entry.get("rewired_str", "--")
#                 improvement = entry.get("improvement", None)
                
#                 base_line.append(base_str)
#                 rewired_line.append(rewired_str)
#                 improv_line.append(f"{improvement:.2f}" if improvement is not None else "--")
#             else:
#                 base_line.append("--")
#                 rewired_line.append("--")
#                 improv_line.append("--")
        
#         # Convert to row strings
#         row_base = " & ".join(base_line) + r" \\"
#         row_rewired = " & ".join(rewired_line) + r" \\"
#         row_improv = " & ".join(improv_line) + r" \\"
        
#         table_rows.extend([row_base, row_rewired, row_improv, r"\midrule"])
    
#     table_footer = r"""
# \bottomrule
# \end{tabular}}
# \end{table}
# """
    
#     table_content = table_header + "\n".join(table_rows) + table_footer
#     return table_content

# # ------------------------------------------------------------------------------
# # Main entry point example
# # ------------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Suppose you have folders like:
#     #   experiments/
#     #       BaseGCN/
#     #           cora/results.json
#     #           citeseer/results.json
#     #           pubmed/results.json
#     #           ...
#     #       HighPassGCN/
#     #           cora/results.json
#     #           ...
#     # or whichever structure you actually use.
    
#     # Let's say we have two such folders: "experiments/BaseGCN" and "experiments/HighPassGCN"
#     model_folders = [
#         ('High-Pass GCN (dir)','rewiring_results/correct_normalisation_local_homophily_with_hp_no_self_loop_2025-02-28-14:25:36/asym'),
#         ('High-Pass GCN (sym)','rewiring_results/correct_normalisation_local_homophily_with_hp_no_self_loop_2025-02-28-14:25:36/sym'),
#         ('Low-Pass GCN (dir)','rewiring_results/correct_normalisation_local_homophily_no_hp_no_self_loop_old_code_2025-02-26-11:33:42/asym'),
#         ('Low-Pass GCN (sym)','rewiring_results/correct_normalisation_local_homophily_no_hp_no_self_loop_old_code_2025-02-26-11:33:42/sym'),
#         # ('High-Pass GCN','rewiring_results/synthetic_datasets_with_hp_2025-02-22-08:53:54/sym'),
#         # ('Low-Pass GCN','rewiring_results/synthetic_graphs_corrected_d_no_hp_2025-02-20-13:31:54/sym')
#         # Add as many as you want...
#     ]
    
#     # Parse results for each folder
#     models_data = {}
#     for display_name, folder in model_folders:
#         models_data[display_name] = parse_results_for_model(folder)

#     # Build the table for main datasets
#     latex_main = build_main_table(
#         models_data,
#         model_order=[m[0] for m in model_folders],  # maintain input order
#         caption="Model Performance Before and After Rewiring (Cora, Citeseer, Pubmed)",
#         label="tab:accuracies"
#     )
    
#     # Build the table for other datasets
#     latex_other = build_other_table(
#         models_data,
#         model_order=[m[0] for m in model_folders],
#         caption="Model Performance Before and After Rewiring (Other Datasets)",
#         label="tab:accuracies_other"
#     )
    
#     # Print or save the resulting LaTeX
#     # (You could print to stdout, or write to file, etc.)
#     print(latex_main)
#     print()
#     print(latex_other)

import os
import json
import math
from pathlib import Path

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MAIN_DATASETS = ["cora", "citeseer", "pubmed",'actor', 'squirrel', 'chameleon']
OTHER_DATASETS = ['wisconsin', 'cornell', 'texas']
SYNTHETIC_DATASETS = [
    'synthetic_graph_dataset_h=0.35_d=10.00',
    'synthetic_graph_dataset_h=0.40_d=10.00',
    'synthetic_graph_dataset_h=0.45_d=10.00',
    'synthetic_graph_dataset_h=0.50_d=10.00',
    'synthetic_graph_dataset_h=0.55_d=10.00',
    'synthetic_graph_dataset_h=0.60_d=10.00',
    'synthetic_graph_dataset_h=0.65_d=10.00'
]

import os
import json
import math
from pathlib import Path
from scipy.stats import t  # add this import

# number of independent trials per condition
N_TRIALS = 10
def parse_single_dataset_results(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # means (as %)
    base_mean    = 100 * data["base_gcn"]["test_accuracy"]
    rewired_mean = 100 * data["rewiring"]["test_accuracy_mean"]

    # CI endpoints
    base_low, base_high       = data["base_gcn"]["test_accuracy_ci"]
    rewired_low, rewired_high = data["rewiring"]["test_accuracy_ci"]

    # half‐ranges of the 95% CI
    base_half    = (base_high - base_low)       * 100 / 2.0
    rewired_half = (rewired_high - rewired_low) * 100 / 2.0

    # approximate sample std dev from CI: half_range = t_{0.975,df} * (s/√n)
    df    = N_TRIALS - 1
    tcrit = t.ppf(0.975, df)
    s_base    = base_half    * math.sqrt(N_TRIALS) / tcrit
    s_rewired = rewired_half * math.sqrt(N_TRIALS) / tcrit

    # Welch’s t‐statistic
    se_diff  = math.sqrt(s_base**2/N_TRIALS + s_rewired**2/N_TRIALS)
    t_stat   = (rewired_mean - base_mean) / se_diff

    # Welch–Satterthwaite df
    num   = (s_base**2/N_TRIALS + s_rewired**2/N_TRIALS)**2
    denom = (
        s_base**4   / ((N_TRIALS**2)*(N_TRIALS-1))
      + s_rewired**4/ ((N_TRIALS**2)*(N_TRIALS-1))
    )
    df_welch = num / denom

    # one‐sided p‐value (rewired > base)
    p_value = 1 - t.cdf(t_stat, df_welch)

    return {
        "base_acc":       base_mean,
        "base_err":       s_base,
        "rewired_acc":    rewired_mean,
        "rewired_err":    s_rewired,
        "p_value":        p_value,
        "is_significant": (p_value < 0.1 and t_stat > 0)
    }

# ------------------------------------------------------------------------------
# Parse results for an entire folder
# ------------------------------------------------------------------------------
def parse_results_for_model(folder_path):
    """
    Parse all available datasets in a folder
    """
    results_dict = {}
    folder_path = Path(folder_path)
    
    for json_file in folder_path.iterdir():
        if json_file.exists() and os.path.basename(json_file).endswith('_results.json') and "all_results" not in os.path.basename(json_file):
            dataset_name = os.path.splitext(os.path.basename(json_file))[0].replace('_results', '').replace('_sym', '').lower()
            results_dict[dataset_name] = parse_single_dataset_results(json_file)
    
    return results_dict

# ------------------------------------------------------------------------------
# Build LaTeX tables
# ------------------------------------------------------------------------------
# ---------------------------------------------------------------------
# NEW: map p-value to LaTeX colour
# ---------------------------------------------------------------------
def p_to_colour(p):
    """
    Blue   : 0.10 > p ≥ 0.05
    Purple : 0.05 > p ≥ 0.01
    Red    : 0.01 > p
    None   : p ≥ 0.10   (no colouring)
    """
    if p is None:
        return None
    if p < 0.01:
        return "red"
    elif p < 0.05:
        return "purple"           # xcolor’s 'purple'
    elif p < 0.10:
        return "blue"
    return None

# ---------------------------------------------------------------------
# REPLACE the old `format_value` with this one
# ---------------------------------------------------------------------
def format_value(value, error, p_value=None, bold=False):
    """
    Build the “mean ± err” string, colour-coding with xcolor and (optionally)
    bold-facing when a result is significant.
    """
    if value is None or math.isnan(value):
        return "--"

    s = f"{value:.2f} ± {error:.2f}"
    
    # colour by p-value threshold
    colour = p_to_colour(p_value)
    if colour:
        s = f"\\textcolor{{{colour}}}{{{s}}}"
    
    # bold if you still want to highlight p < 0.05
    if bold:
        s = f"\\textbf{{{s}}}"
    
    return s
# ------------------------------------------------------------------------------
# Build a LaTeX table for the given datasets
# ------------------------------------------------------------------------------

def build_table(models_data, dataset_list, caption, label, columns):
    """Build a LaTeX table for the given datasets"""
    column_spec = "l" + "c" * len(dataset_list)
    header_columns = " & ".join([col for col in columns])
    
    table_header = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\makebox[\\textwidth]{{\\begin{{tabular}}{{{column_spec}}}
\\toprule
Model & {header_columns} \\\\
\\midrule
"""
    
    table_rows = []
    for model_name, model_data in models_data.items():
        base_row = [f"{model_name} (Base)"]
        rewired_row = [f"{model_name} (Rewired)"]
        for dataset in dataset_list:
            entry = model_data.get(dataset, {})
            print(dataset,model_data)
            if entry:
                # ----- Base (never coloured) -----
                base_row.append(
                    format_value(
                        entry.get("base_acc"),
                        entry.get("base_err")
                    )
                )

                # ----- Rewired (colour & bold) -----
                rewired_row.append(
                    format_value(
                        entry.get("rewired_acc"),
                        entry.get("rewired_err"),
                        p_value = entry.get("p_value"),
                        #bold    = entry.get("p_value", 1) < 0.05  # bold when significant
                    )
                )
            else:
                base_row.append("--")
                rewired_row.append("--")
        
        # Convert to row strings
        table_rows.append(" & ".join(base_row) + r" \\")
        table_rows.append(" & ".join(rewired_row) + r" \\")
        table_rows.append(r"\midrule")
    
    table_footer = r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    
    table_content = table_header + "\n".join(table_rows) + table_footer
    return table_content

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():
    # Define model folders - adjust these paths to your actual data locations
    model_folders = [
        #('High-Pass GCN', 'results/rewiring/no_self_loop_yes_hp_fixed_n_layers_2025-03-20-09-32-06'),
        # ('Single Rewiring', 'results/rewiring/large_run_no_hp_2025-05-07-14-01-06'),
        # ('Short Rewiring', 'results/rewiring/incremental_iterative_selective_rewiring_full_2025-05-02-13-04-15'),
        # ('Iterative Rewiring', 'results/rewiring/long_rewiring_download'),
        # ('Full Block Rewiring', 'results/rewiring/full_block_rewiring_2025-05-12-14-11-54')
        ('GCN','results/rewiring/final_synthetic_data_collection')
    ]
    
    # Parse results for each folder
    models_data = {}
    for display_name, folder in model_folders:
        models_data[display_name] = parse_results_for_model(folder)
    print(models_data)
    # Build tables
    main_table = build_table(
        models_data,
        MAIN_DATASETS,
        "Model Performance Before and After Rewiring",
        "tab:accuracies",
        ["Cora", "Citeseer", "Pubmed", "Actor", "Squirrel", "Chameleon"]
    )
    
    other_table = build_table(
        models_data,
        OTHER_DATASETS,
        "Model Performance Before and After Rewiring (Other Datasets)",
        "tab:accuracies_other",
        ["Actor", "Squirrel", "Chameleon", "Wisconsin", "Cornell", "Texas"]
    )
    
    synthetic_table = build_table(
        models_data,
        SYNTHETIC_DATASETS,
        "Model Performance Before and After Rewiring (Planted Partition SBM Datasets)",
        "tab:accuracies_synthetic",
        ["h=0.35", "h=0.40", "h=0.45", "h=0.50", "h=0.55", "h=0.60", "h=0.65"]
    )
    
    # Print the resulting LaTeX
    print(main_table)
    print(other_table)
    print(synthetic_table)
    
    # Optionally save to files
    with open("main_table.tex", "w") as f:
        f.write(main_table)
    
    with open("other_table.tex", "w") as f:
        f.write(other_table)
        
    with open("synthetic_table.tex", "w") as f:
        f.write(synthetic_table)

if __name__ == "__main__":
    main()
