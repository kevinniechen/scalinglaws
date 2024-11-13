import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt

# Subset of tasks for avg_subset metric
SUBSET_TASKS = [
    "bigbench_operators", "pubmed_qa_labeled", "hellaswag_zeroshot",
    "boolq", "arc_easy", "coqa", "bigbench_dyck_languages",
    "lambada_openai", "bigbench_novel_concepts", "winograd",
    "bigbench_cs_algorithms", "commonsense_qa", "bigbench_qa_wikidata",
    "hellaswag", "copa", "squad", "piqa"
]

def load_model_data(model_json_path, cc_mults, datasets, eval_dir=None):
    """Load and parse a single model's data"""
    with open(model_json_path) as f:
        data = json.load(f)
        
    cc_mult = data["hyperparameters"]["chinchilla_multiplier"]
    dataset_name = data["dataset_name"]
    
    if cc_mult not in cc_mults or dataset_name not in datasets:
        return None
        
    model = {
        "cc_mult": cc_mult,
        "dataset_name": dataset_name,
        "name": data["name"],
        "model_name": data["hyperparameters"]["model"].split("/")[-1].split(".")[0],
        "N": data["hyperparameters"]["params"],
        "D": data["hyperparameters"]["tokens"],
        "tok_mult": cc_mult * 20
    }
    
    # Add loss metrics
    for result in data["results"]:
        suffix = result["val_data"][0].split("/")[-2]
        if "de-en" in suffix:
            suffix = result["val_data"][0].split("/")[-1].split(".")[0]
        model[f"loss_{suffix}"] = result["loss"]
    
    # Add evaluation metrics if available
    if eval_dir:
        eval_path = f"{eval_dir}/evaluation_{Path(model_json_path).stem}_heavy.json"
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                eval_data = json.load(f)
                metrics = eval_data["eval_metrics"]["icl"]
                
                # Store individual task errors
                for k, v in metrics.items():
                    model[f"err_{k}"] = 1.0 - v
                
                # Calculate averages
                model["err_avg"] = np.mean([1.0 - v for v in metrics.values()])
                subset_metrics = [1.0 - metrics[k] for k in SUBSET_TASKS if k in metrics]
                if subset_metrics:
                    model["err_avg_subset"] = np.mean(subset_metrics)
                
    return model

def load_scaling_laws():
    """Load pre-fitted scaling law parameters"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(base_dir, "scaling/scaling_law_dict.pkl")
    
    with open(pickle_path, 'rb') as f:
        scaling_laws = pickle.load(f)
        return scaling_laws

def exp_scaling(L, a, b, e):
    """Exponential decay function for error scaling - matches decay_ours from laws.py"""
    return e - a * np.exp(L) ** (-b)

def fit_error_scaling(loss, error):
    """Fit error scaling with exponential decay model - exact match to curve_fit_decay_ours"""
    try:
        # Match laws.py exactly - no initial guesses, no grid search
        popt, _ = curve_fit(
            exp_scaling,
            loss,
            error,
            maxfev=10000
        )
        return popt

    except Exception as e:
        print(f"Error fitting exponential: {str(e)}")
        return None

def analyze_all_scalings(datasets, val_dataset, downstream, model_dir, eval_dir, cc_mults, scaling_laws):
    """Create subplots for each dataset in a single figure"""
    # Create a figure with subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, dataset in enumerate(datasets):
        # Load models for this dataset
        models = []
        for filename in os.listdir(model_dir):
            if filename.endswith('.json'):
                model = load_model_data(f"{model_dir}/{filename}", cc_mults, [dataset], eval_dir)
                if model and f"err_{downstream}" in model:
                    models.append(model)
        
        if not models:
            print(f"No valid models found for {dataset} -> {downstream}")
            continue
        
        df = pd.DataFrame(models)
        ax = axes[idx]  # Get the appropriate subplot
        
        try:
            scaling_key = f"train={dataset}-loss={val_dataset}-downstream={downstream}"
            scaling_data = scaling_laws[scaling_key]
            error_params = scaling_data['error_scaling']
            
            error = df[f'err_{downstream}'].values
            loss = df[f'loss_{val_dataset}'].values
            
            # Fit our custom parameters
            exp_params = fit_error_scaling(loss, error)
            
            # Plot raw data points
            ax.scatter(loss, error, color='blue', label='Data points', alpha=0.6, marker='o')
            
            # Generate smooth curve for both fits
            loss_smooth = np.linspace(min(loss), max(loss), 100)
            
            # Plot pre-fitted curve
            y_prefitted = exp_scaling(loss_smooth, *error_params)
            ax.plot(loss_smooth, y_prefitted, 'r-', label='Pre-fitted', alpha=0.8)
            
            # Plot our custom fit
            if exp_params is not None:
                y_custom = exp_scaling(loss_smooth, *exp_params)
                ax.plot(loss_smooth, y_custom, 'g--', label='Custom fit', alpha=0.8)
            
            # Configure subplot
            ax.set_xlabel('Loss')
            ax.set_ylabel('Error')
            ax.set_title(f'{dataset}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Print parameters
            error_params_fmt = [f"{x:.4f}" for x in error_params]
            exp_params_fmt = [f"{x:.4f}" for x in exp_params] if exp_params is not None else ["None"]
            
            print(f"\nScaling laws for {dataset} ({val_dataset} validation, {downstream} downstream):")
            print(f"Pre-fitted error exponential fit (a, b, e):               {', '.join(error_params_fmt)}")
            print(f"Custom error exponential fit (a, b, e):        {', '.join(exp_params_fmt)}")
            
        except Exception as e:
            print(f"Error analyzing scaling laws for {dataset} -> {downstream}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Add overall title
    fig.suptitle(f'Error Scaling Comparison (downstream={downstream})', y=1.05)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save combined plot
    plt.savefig(f'scaling_comparison_combined_{downstream}.png', bbox_inches='tight')
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "scaling/exp_data/models")
    eval_dir = os.path.join(base_dir, "scaling/exp_data/evals")
    
    scaling_laws = load_scaling_laws()
    datasets = ["c4_original", "rpj", "rw_original"]
    cc_mults = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    val_dataset = "c4_val"
    
    for downstream in ["avg", "avg_subset"]:
        analyze_all_scalings(
            datasets, val_dataset, downstream,
            model_dir, eval_dir, cc_mults, scaling_laws
        )

if __name__ == "__main__":
    main()