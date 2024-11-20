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

# Subset of tasks for avg_best_subset metric
BEST_SUBSET_TASKS = [
    "winograd", "copa", "piqa", "hellaswag", "hellaswag_zeroshot",
    "arc_easy", "bigbench_qa_wikidata", "boolq", "winogrande",
    "lambada_openai"
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

                best_subset_metrics = [1.0 - metrics[k] for k in BEST_SUBSET_TASKS if k in metrics]
                if best_subset_metrics:
                    model["err_avg_best_subset"] = np.mean(best_subset_metrics)
                
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

def sigmoid_scaling(L, a, b, c, d):
    """Sigmoidal function for error scaling"""
    return d + (a - d) / (1 + np.exp(-b * (L - c)))

def fit_sigmoid_scaling(loss, error):
    """Fit error scaling with sigmoid model"""
    try:
        # Initial parameter guesses for sigmoid
        # a: upper asymptote (max error)
        # b: steepness
        # c: midpoint (around mean loss)
        # d: lower asymptote (min error)
        p0 = [
            np.max(error),  # a: upper asymptote
            1.0,           # b: steepness
            np.mean(loss), # c: midpoint
            np.min(error)  # d: lower asymptote
        ]
        
        # Bounds to keep parameters in reasonable ranges
        bounds = (
            [0, 0, np.min(loss), 0],  # lower bounds
            [2, 10, np.max(loss), 1]   # upper bounds
        )
        
        popt, _ = curve_fit(
            sigmoid_scaling,
            loss,
            error,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        return popt
    except Exception as e:
        print(f"Error fitting sigmoid: {str(e)}")
        return None

def zero_to_one_sigmoid_scaling(L, b, c):
    """Sigmoidal function for mapping log reducible loss to error"""
    return 1 / (1 + np.exp(-b * (L - c)))

def fit_zero_to_one_sigmoid_scaling(loss, error):
    """Fit error scaling with sigmoid model"""
    try:
        # Initial parameter guesses for sigmoid
        # b: steepness
        # c: midpoint (around mean loss)
        p0 = [
            1.0,           # b: steepness
            np.mean(loss), # c: midpoint
        ]
        
        # Bounds to keep parameters in reasonable ranges
        bounds = (
            [-100, -10],  # lower bounds
            [100, 10]   # upper bounds
        )
        
        popt, _ = curve_fit(
            zero_to_one_sigmoid_scaling,
            loss,
            error,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        return popt
    except Exception as e:
        print(f"Error fitting zero to one sigmoid: {str(e)}")
        return None

def analyze_all_scalings(datasets, val_dataset, downstream, model_dir, eval_dir, cc_mults, scaling_laws):
    """Create subplots for each dataset with both sigmoid and exponential fits"""
    # Create single figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
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
        ax = axes[0, idx]  # Get the appropriate subplot
        log_ax = axes[1, idx]
        
        try:
            if downstream == "avg_best_subset":
                scaling_key = f"train={dataset}-loss={val_dataset}-downstream=avg_subset"
            else:
                scaling_key = f"train={dataset}-loss={val_dataset}-downstream={downstream}"
            scaling_data = scaling_laws[scaling_key]
            error_params = scaling_data['error_scaling']
            loss_scaling = scaling_data['loss_scaling']
            irr_loss = loss_scaling[3] # also called E in the paper/code
            
            # Print out tasks by error rate
            errs = []
            for k in df.keys():
                if "err_" in k:
                    errs.append((df[k].values.min(), k))
            
            sorted_errs = sorted(errs, key = lambda x: x[0])
            print(sorted_errs)


            # Calculate errors
            error = df[f'err_{downstream}'].values
            accuracy = 1-error
            loss = df[f'loss_{val_dataset}'].values
            reducible_loss = loss - irr_loss
            log_reducible_loss = -np.log(reducible_loss)

            indices = log_reducible_loss > -1
            log_reducible_loss = log_reducible_loss[indices]
            accuracy = accuracy[indices]
            
            # Fit our custom parameters
            exp_params = fit_error_scaling(loss, error)
            sig_params = fit_sigmoid_scaling(loss, error)
            log_sig_params = fit_zero_to_one_sigmoid_scaling(log_reducible_loss, accuracy)
            
            # Generate extended range for smooth curves to see full behavior
            loss_min = min(loss)
            loss_max = max(loss)
            loss_range = loss_max - loss_min

            log_loss_min = min(log_reducible_loss)
            log_loss_max = max(log_reducible_loss)
            log_loss_range = log_loss_max - log_loss_min
            
            # Extend the range by 50% on both sides
            loss_smooth = np.linspace(
                loss_min - 0.5 * loss_range,
                loss_max + 0.5 * loss_range,
                200
            )

            # Extend range more to the left to see trend for log loss
            log_loss_smooth = np.linspace(
                log_loss_min - 0.5 * log_loss_range,
                log_loss_max + 2 * log_loss_range,
                200
            )
            
            # Plot data points
            ax.scatter(loss, error, color='blue', label='Data points', alpha=0.6, marker='o')
            log_ax.set_ylim([-0.1, 1.1])
            log_ax.set_xlim([log_loss_min - 0.5 * log_loss_range, log_loss_max + 2 * log_loss_range])
            log_ax.scatter(log_reducible_loss, accuracy, color='blue', label='Data points', alpha=0.6, marker='o')
            
            # Add vertical lines to show data range
            ax.axvline(x=loss_min, color='gray', linestyle=':', alpha=0.3)
            ax.axvline(x=loss_max, color='gray', linestyle=':', alpha=0.3)
            
            # Plot pre-fitted exponential curve
            y_prefitted = exp_scaling(loss_smooth, *error_params)
            ax.plot(loss_smooth, y_prefitted, 'r-', label='Pre-fitted exp', alpha=0.8)
            
            # Plot our custom exponential fit
            if exp_params is not None:
                y_custom = exp_scaling(loss_smooth, *exp_params)
                ax.plot(loss_smooth, y_custom, 'g--', label='Custom exp', alpha=0.8)
            
            # Plot our custom sigmoid fit
            if sig_params is not None:
                y_sigmoid = sigmoid_scaling(loss_smooth, *sig_params)
                ax.plot(loss_smooth, y_sigmoid, 'm:', label='Sigmoid', alpha=0.8, linewidth=2)

            # Plot our log sigmoid fit
            if log_sig_params is not None:
                y_sigmoid = zero_to_one_sigmoid_scaling(log_loss_smooth, *log_sig_params)
                log_ax.plot(log_loss_smooth, y_sigmoid, 'm:', label='Sigmoid', alpha=0.8, linewidth=2)
            
            # Configure subplot
            ax.set_xlabel('Loss')
            ax.set_ylabel('Error')
            ax.set_title(f'{dataset}\nData range shown in vertical lines')
            ax.legend()
            ax.grid(True, alpha=0.3)

            log_ax.set_xlabel('Negative Log Reducible Loss')
            log_ax.set_ylabel('Accuracy')
            log_ax.legend()
            log_ax.grid(True, alpha=0.3)
            
            # Print parameters and ranges
            print(f"\nScaling laws for {dataset}:")
            print(f"Loss range: [{loss_min:.4f}, {loss_max:.4f}]")
            print(f"Error range: [{min(error):.4f}, {max(error):.4f}]")
            print(f"Pre-fitted error exponential fit (a, b, e):               {', '.join([f'{x:.4f}' for x in error_params])}")
            print(f"Custom error exponential fit (a, b, e):        {', '.join([f'{x:.4f}' for x in exp_params]) if exp_params is not None else 'None'}")
            print(f"Custom error sigmoid fit (a, b, c, d):         {', '.join([f'{x:.4f}' for x in sig_params]) if sig_params is not None else 'None'}")
            print(f"Log loss error sigmoid fit (b, c):         {', '.join([f'{x:.4f}' for x in log_sig_params]) if log_sig_params is not None else 'None'}")
            
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
    
    for downstream in ["avg", "avg_subset", "avg_best_subset"]:
        analyze_all_scalings(
            datasets, val_dataset, downstream,
            model_dir, eval_dir, cc_mults, scaling_laws
        )

if __name__ == "__main__":
    main()