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

def exp_scaling(log_L, a, b, e):
    """Exponential decay function for error scaling using log of irreducible loss"""
    return e - a * np.exp(log_L) ** (-b)

def fit_error_scaling(loss, error):
    """Fit error scaling with exponential decay model using log of loss"""
    try:
        # Take log of loss values
        log_loss = np.log(loss)
        
        popt, _ = curve_fit(
            exp_scaling,
            log_loss,
            error,
            maxfev=10000
        )
        return popt

    except Exception as e:
        print(f"Error fitting exponential: {str(e)}")
        return None

def sigmoid_scaling(log_L, a, b, c, d):
    """Sigmoidal function for error scaling using log of irreducible loss"""
    return d + (a - d) / (1 + np.exp(b * (log_L - c)))

def fit_sigmoid_scaling(loss, error):
    """Fit error scaling with sigmoid model using log of loss"""
    try:
        log_loss = np.log(loss)
        
        # Initial parameter guesses for sigmoid
        p0 = [
            np.max(error),    # a: upper asymptote at max error
            3.0,              # b: moderate initial steepness
            np.mean(log_loss),# c: midpoint at mean of log loss
            np.min(error)     # d: lower asymptote at min error
        ]
        
        # Much looser bounds to allow better fitting
        bounds = (
            [0.0, 0.1, -np.inf, 0.0],     # lower bounds
            [1.0, 100.0, np.inf, 1.0]      # upper bounds
        )
        
        popt, _ = curve_fit(
            sigmoid_scaling,
            log_loss,
            error,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        return popt
    except Exception as e:
        print(f"Error fitting sigmoid: {str(e)}")
        return None

def fit_raw_loss_curve(N, loss):
    """Fit raw loss curve to find irreducible loss (asymptote)"""
    def loss_scaling(N, alpha, beta, L_min):
        # Basic power law with asymptote
        return L_min + alpha * (N ** -beta)
    
    try:
        # Initial guesses
        p0 = [1.0, 0.5, np.min(loss)]
        
        # Bounds to ensure positive values and reasonable asymptote
        bounds = (
            [0, 0, 0],  # lower bounds
            [np.inf, 2, np.min(loss) * 1.2]  # upper bounds
        )
        
        popt, _ = curve_fit(
            loss_scaling,
            N,
            loss,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        return popt
    except Exception as e:
        print(f"Error fitting raw loss curve: {str(e)}")
        return None

def analyze_all_scalings(datasets, val_dataset, downstream, model_dir, eval_dir, cc_mults, scaling_laws):
    """Create subplots for each dataset with both sigmoid and exponential fits"""
    # Create single figure with subplots
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
            raw_loss = df[f'loss_{val_dataset}'].values
            
            # Get irreducible loss from pre-fitted parameters
            L_irr = error_params[2]  # The 'e' parameter from the pre-fitted curve
            print(f"\nIrreducible loss for {dataset}: {L_irr:.4f}")
            
            # Calculate reducible loss
            L_red = raw_loss - L_irr
            
            # Calculate min and max values for plotting
            L_red_min = np.min(L_red)
            L_red_max = np.max(L_red)
            
            # Fit our custom parameters using log of reducible loss
            exp_params = fit_error_scaling(L_red, error)
            sig_params = fit_sigmoid_scaling(L_red, error)
            
            # Generate extended range for smooth curves
            log_L_red_min = np.log(L_red_min)
            log_L_red_max = np.log(L_red_max)
            log_L_red_range = log_L_red_max - log_L_red_min
            
            # Extend the range by 50% on both sides in log space
            log_L_red_smooth = np.linspace(
                log_L_red_min - 0.5 * log_L_red_range,
                log_L_red_max + 0.5 * log_L_red_range,
                200
            )
            
            # Convert back to normal space for plotting
            L_red_smooth = np.exp(log_L_red_smooth)
            loss_smooth = L_red_smooth + L_irr  # Add back irreducible loss for plotting
            
            # Plot data points using raw loss for x-axis
            ax.scatter(raw_loss, error, color='blue', label='Data points', alpha=0.6, marker='o')
            
            # Add vertical lines to show data range
            ax.axvline(x=L_red_min + L_irr, color='gray', linestyle=':', alpha=0.3)
            ax.axvline(x=L_red_max + L_irr, color='gray', linestyle=':', alpha=0.3)
            
            # Plot pre-fitted exponential curve using original function
            y_prefitted = error_params[2] - error_params[0] * np.exp(np.log(L_red_smooth)) ** (-error_params[1])
            ax.plot(loss_smooth, y_prefitted, 'r-', label='Pre-fitted exp', alpha=0.8)
            
            # Plot our custom exponential fit
            if exp_params is not None:
                y_custom = exp_scaling(log_L_red_smooth, *exp_params)
                ax.plot(loss_smooth, y_custom, 'g--', label='Custom exp', alpha=0.8)
            
            # Plot our custom sigmoid fit
            if sig_params is not None:
                y_sigmoid = sigmoid_scaling(log_L_red_smooth, *sig_params)
                ax.plot(loss_smooth, y_sigmoid, 'm:', label='Sigmoid', alpha=0.8, linewidth=2)
            
            # Configure subplot with log scale on x-axis
            ax.set_xscale('log')
            ax.set_xlabel('Loss (log scale)')
            ax.set_ylabel('Error')
            ax.set_title(f'{dataset}\nData range shown in vertical lines')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Print parameters and ranges
            print(f"\nScaling laws for {dataset}:")
            print(f"Error range: [{min(error):.4f}, {max(error):.4f}]")
            print(f"Pre-fitted error exponential fit (a, b, e):               {', '.join([f'{x:.4f}' for x in error_params])}")
            print(f"Custom error exponential fit (a, b, e):        {', '.join([f'{x:.4f}' for x in exp_params]) if exp_params is not None else 'None'}")
            print(f"Custom error sigmoid fit (a, b, c, d):         {', '.join([f'{x:.4f}' for x in sig_params]) if sig_params is not None else 'None'}")
            
        except Exception as e:
            print(f"Error analyzing scaling laws for {dataset} -> {downstream}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Add overall title
    fig.suptitle(f'Error Scaling Comparison (downstream={downstream})', y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'scaling_comparison_{downstream}.png', bbox_inches='tight')
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