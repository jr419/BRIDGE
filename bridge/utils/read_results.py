import os
import json
from pathlib import Path
from textwrap import indent

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MAIN_DATASETS = ["cora", "citeseer", "pubmed"]
OTHER_DATASETS = ['actor','squirrel','chameleon','wisconsin','cornell','texas']
# OTHER_DATASETS = ['synthetic_graph_dataset_h=0.10_d=20.00',
#                   'synthetic_graph_dataset_h=0.20_d=20.00',
#                   'synthetic_graph_dataset_h=0.30_d=20.00',
#                   'synthetic_graph_dataset_h=0.40_d=20.00',
#                   'synthetic_graph_dataset_h=0.50_d=20.00',
#                   'synthetic_graph_dataset_h=0.60_d=20.00',
#                   'synthetic_graph_dataset_h=0.70_d=20.00'
#                 ]
                  
# If your dataset names have different cases or underscores,
# adjust them here as needed. Also, if your folder structure differs,
# you can change how parse_results finds and reads them.

RESULTS_FILENAME = "results.json"

# ------------------------------------------------------------------------------
# Helper to parse a single dataset's JSON file
# ------------------------------------------------------------------------------

def parse_single_dataset_results(json_path):
    """
    Reads a single JSON results file and returns a dict with:
      {
        'base_acc': float,
        'rewired_acc': float,
        'improvement': float
      }
    using the test_accuracy (or test_accuracy_mean) from your JSON structure.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # For example, from your sample JSON, we can extract:
    base_acc = 100*data["base_gcn"]["test_accuracy"]  # test_accuracy
    base_acc_ci = data["base_gcn"]["test_accuracy_ci"]
    # For "rewiring", if there's a mean and confidence interval, we can either
    # use the mean or store the full ± CI. We'll store the mean plus the ± from CI.
    rewired_mean = 100*data["rewiring"]["test_accuracy_mean"]
    # If you have a test_accuracy_ci, we can format that as well:
    rewired_acc_ci = data["rewiring"]["test_accuracy_ci"]  # [low, high]

    # A simple way to format is "mean ± std", 
    # but if you want to store the exact CI in LaTeX, you can do that too.
    # Here, we’ll do something like "mean ± half_range":
    base_half_range = 100*(base_acc_ci[1] - base_acc_ci[0]) / 2.0
    rewired_half_range = 100*(rewired_acc_ci[1] - rewired_acc_ci[0]) / 2.0
    rewired_str = f"{rewired_mean:.2f} ± {rewired_half_range:.2f}"
    base_str = f"{base_acc:.2f} ± {base_half_range:.2f}"
    
    # Or if you want the raw mean as a float:
    rewired_acc = rewired_mean
    
    # improvement percentage (already in your JSON as improvement_percentage?)
    improvement = data["improvement_percentage"]
    
    return {
        "base_acc": base_acc,
        "rewired_acc": rewired_acc,
        "base_str": base_str,  # store the string for LaTeX
        "rewired_str": rewired_str,  # store the string for LaTeX
        "improvement": improvement,
    }

# ------------------------------------------------------------------------------
# Parse results for an entire folder (i.e., one "model family")
# ------------------------------------------------------------------------------

def parse_results_for_model(folder_path):
    """
    Given a folder_path which contains subfolders or files for each dataset
    (e.g. "cora/results.json", "citeseer/results.json", etc.),
    parse all available datasets and return a dictionary:
        {
            'cora': {...},
            'citeseer': {...},
            'pubmed': {...},
            'actor': {...},
            ...
        }
    where each value is a dict with 'base_acc', 'rewired_acc', and 'improvement'.
    """
    results_dict = {}
    folder_path = Path(folder_path)
    
    # We assume each dataset is in a subfolder named exactly "cora", "citeseer", etc.
    # If, instead, you have something like "cora.json" directly in folder_path,
    # you can adapt the code to look for JSON files matching dataset names.
    
    for json_file in folder_path.iterdir():
        if json_file.exists() and os.path.basename(json_file).endswith('_results.json'):
            dataset_name = os.path.splitext(os.path.basename(json_file))[0].replace('_results', '').replace('_sym', '').lower()  # e.g. "cora", "citeseer"
            print(dataset_name)
            results_dict[dataset_name] = parse_single_dataset_results(json_file)
    
    return results_dict

# ------------------------------------------------------------------------------
# Build LaTeX table for main datasets (Cora, Citeseer, Pubmed)
# ------------------------------------------------------------------------------
def build_main_table(models_data, model_order=None, caption="", label=""):
    """
    models_data is a dict:
       {
         'ModelDisplayName1': {'cora': {...}, 'citeseer': {...}, ...},
         'ModelDisplayName2': {...}
       }
    model_order is a list specifying the order of model rows in the table.
    We produce a LaTeX table with columns: Model, Cora, Citeseer, Pubmed
    and row groups: (Base, Rewired, Improvement) for each model.
    
    We'll assume that each model_data dict has "cora", "citeseer", "pubmed" keys
    with the standard sub-keys: 'base_acc', 'rewired_str', 'improvement', etc.
    """
    if model_order is None:
        model_order = list(models_data.keys())
    
    # You can adapt columns to your needs
    table_header = r"""
\begin{table}[htbp]
\centering
\caption{%s}
\label{%s}
\makebox[\textwidth]{\begin{tabular}{lccc}
\toprule
Model & Cora & Citeseer & Pubmed \\
\midrule
""" % (caption, label)
    
    table_rows = []
    
    for model_name in model_order:
        # The dict for this model
        md = models_data[model_name]
        print(md)
        # model_name = model_name.replace('High-Pass','Low-Pass')
        
        # For each dataset in MAIN_DATASETS, fetch the data.
        cora = md.get("cora", {})
        citeseer = md.get("citeseer", {})
        pubmed = md.get("pubmed", {})
        
        # If a dataset is missing, you could handle that gracefully
        # (e.g. put --)
        def format_base_acc(x):
            return x["base_str"] if x else "--"
        
        def format_rewired_acc_str(x):
            return x["rewired_str"] if x else "--"
        
        def format_improvement(x):
            return f"{x['improvement']:.2f}" if x else "--"
        
        # row 1: Base
        row_base = (
            f"{model_name} (Base) & " 
            f"{format_base_acc(cora)} & "
            f"{format_base_acc(citeseer)} & "
            f"{format_base_acc(pubmed)} \\\\"
        )
        
        # row 2: Rewired
        row_rewired = (
            f"{model_name} (Rewired) & "
            f"{format_rewired_acc_str(cora)} & "
            f"{format_rewired_acc_str(citeseer)} & "
            f"{format_rewired_acc_str(pubmed)} \\\\"
        )
        
        # row 3: Improvement
        row_improvement = (
            f"{model_name} (Improvement \\%) & "
            f"{format_improvement(cora)} & "
            f"{format_improvement(citeseer)} & "
            f"{format_improvement(pubmed)} \\\\"
        )
        
        table_rows.extend([row_base, row_rewired, row_improvement, r"\midrule"])
    
    table_footer = r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    
    table_content = table_header + "\n".join(table_rows) + table_footer
    return table_content

# ------------------------------------------------------------------------------
# Build LaTeX table for "other" datasets (everything else)
# ------------------------------------------------------------------------------
def build_other_table(models_data, model_order=None, caption="", label=""):
    """
    Very similar to build_main_table, but we gather columns for everything
    in each model’s dict that is *not* in MAIN_DATASETS.
    Then produce one row per dataset, or produce a wide table with all columns.
    
    In your sample, you show a table with columns for:
      Actor, Chameleon, Squirrel, Texas, Wisconsin, Cornell
    and rows for Base, Rewired, Improvement.
    
    For demonstration, we’ll assume we know the "other" dataset list or
    we can find them from the first model’s data keys.
    """
    if model_order is None:
        model_order = list(models_data.keys())
    
    # Collect all "other" dataset names from the first model
    # (you can unify them across all models if needed).
    all_datasets = set()
    for model_name in model_order:
        all_datasets.update(models_data[model_name].keys())
    # Filter out main
    # Sort them as you like:
    # e.g. ds_order = ["actor", "chameleon", ...] if you want a fixed order:
    ds_order = OTHER_DATASETS
    
    # Build the header line for dataset columns:
    header_columns = " & ".join(name.capitalize() for name in ds_order)
    
    table_header = r"""
\begin{table}[htbp]
\centering
\caption{%s}
\label{%s}
\makebox[\textwidth]{\begin{tabular}{l%s}
\toprule
Model & %s \\
\midrule
""" % (caption, label, "c" * len(ds_order), header_columns)
    
    table_rows = []
    
    for model_name in model_order:
        md = models_data[model_name]
        
        # We'll create three lines: Base, Rewired, Improvement
        # across all "other" dataset columns
        base_line = [f"{model_name} (Base)"]
        rewired_line = [f"{model_name} (Rewired)"]
        improv_line = [f"{model_name} (Improvement \\%)"]
        
        for ds in ds_order:
            
            entry = md.get(ds, {})
            if entry:
                base_str = entry.get("base_str", None)
                rewired_str = entry.get("rewired_str", "--")
                improvement = entry.get("improvement", None)
                
                base_line.append(base_str)
                rewired_line.append(rewired_str)
                improv_line.append(f"{improvement:.2f}" if improvement is not None else "--")
            else:
                base_line.append("--")
                rewired_line.append("--")
                improv_line.append("--")
        
        # Convert to row strings
        row_base = " & ".join(base_line) + r" \\"
        row_rewired = " & ".join(rewired_line) + r" \\"
        row_improv = " & ".join(improv_line) + r" \\"
        
        table_rows.extend([row_base, row_rewired, row_improv, r"\midrule"])
    
    table_footer = r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    
    table_content = table_header + "\n".join(table_rows) + table_footer
    return table_content

# ------------------------------------------------------------------------------
# Main entry point example
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose you have folders like:
    #   experiments/
    #       BaseGCN/
    #           cora/results.json
    #           citeseer/results.json
    #           pubmed/results.json
    #           ...
    #       HighPassGCN/
    #           cora/results.json
    #           ...
    # or whichever structure you actually use.
    
    # Let's say we have two such folders: "experiments/BaseGCN" and "experiments/HighPassGCN"
    model_folders = [
        ('High-Pass GCN (dir)','rewiring_results/correct_normalisation_local_homophily_with_hp_no_self_loop_2025-02-28-14:25:36/asym'),
        ('High-Pass GCN (sym)','rewiring_results/correct_normalisation_local_homophily_with_hp_no_self_loop_2025-02-28-14:25:36/sym'),
        ('Low-Pass GCN (dir)','rewiring_results/correct_normalisation_local_homophily_no_hp_no_self_loop_old_code_2025-02-26-11:33:42/asym'),
        ('Low-Pass GCN (sym)','rewiring_results/correct_normalisation_local_homophily_no_hp_no_self_loop_old_code_2025-02-26-11:33:42/sym'),
        # ('High-Pass GCN','rewiring_results/synthetic_datasets_with_hp_2025-02-22-08:53:54/sym'),
        # ('Low-Pass GCN','rewiring_results/synthetic_graphs_corrected_d_no_hp_2025-02-20-13:31:54/sym')
        # Add as many as you want...
    ]
    
    # Parse results for each folder
    models_data = {}
    for display_name, folder in model_folders:
        models_data[display_name] = parse_results_for_model(folder)

    # Build the table for main datasets
    latex_main = build_main_table(
        models_data,
        model_order=[m[0] for m in model_folders],  # maintain input order
        caption="Model Performance Before and After Rewiring (Cora, Citeseer, Pubmed)",
        label="tab:accuracies"
    )
    
    # Build the table for other datasets
    latex_other = build_other_table(
        models_data,
        model_order=[m[0] for m in model_folders],
        caption="Model Performance Before and After Rewiring (Other Datasets)",
        label="tab:accuracies_other"
    )
    
    # Print or save the resulting LaTeX
    # (You could print to stdout, or write to file, etc.)
    print(latex_main)
    print()
    print(latex_other)
