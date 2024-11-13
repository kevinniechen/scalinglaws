import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import json

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

def try_fit_function(fit_func, x, y, param_ranges, maxfev=10000):
    """Generic function to try multiple initial parameters for any fitting function"""
    min_residual = float("inf")
    best_params = None
    
    # Generate all combinations of initial parameters
    param_combinations = np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(param_ranges))
    
    for p0 in param_combinations:
        try:
            popt, _ = curve_fit(fit_func, x, y, p0=p0, maxfev=maxfev)
            y_fit = fit_func(x, *popt)
            residuals = y - y_fit
            curr_residual = np.sqrt(np.mean(residuals**2))
            
            if curr_residual < min_residual:
                min_residual = curr_residual
                best_params = popt
        except:
            continue
            
    return best_params, min_residual

def fit_loss_scaling(N, M, loss):
    """Fit loss scaling with both sigmoid and power law models"""
    def sigmoid_scaling(x, alpha, beta, b, E):
        N, M = x
        return (alpha / (1 + np.exp(-b * N))) + (beta / (1 + np.exp(-b * M * N))) + E
        
    def power_scaling(x, alpha, beta, b, E):
        N, M = x
        return alpha * np.power(N, b) + beta * np.power(M * N, b) + E
    
    # Try both sigmoid and power law fits
    sigmoid_ranges = [
        [-1e4, -1e3, -1e2],  # alpha
        [1e2, 1e3, 1e4],     # beta
        [1e-4, 1e-3, 1e-2],  # b
        [0.0, 1e2, 1e3]      # E
    ]
    
    power_ranges = [
        [1e2, 1e3, 1e4],     # alpha
        [1e2, 1e3, 1e4],     # beta
        [-0.3, -0.2, -0.1],  # b
        [0.0, 1e2, 1e3]      # E
    ]
    
    sigmoid_params, sigmoid_residual = try_fit_function(
        sigmoid_scaling, (N, M), loss, sigmoid_ranges
    )
    
    power_params, power_residual = try_fit_function(
        power_scaling, (N, M), loss, power_ranges
    )
    
    # Return the better fit
    if sigmoid_residual < power_residual:
        return sigmoid_params, 'sigmoid'
    else:
        return power_params, 'power'

def fit_error_scaling(loss, error):
    """Fit error scaling with exponential decay"""
    def error_scaling(L, k, gamma, epsilon):
        return epsilon - k * np.exp(-gamma * L)
    
    param_ranges = [
        [1.0, 2.0, 3.0],     # k (amplitude)
        [0.5, 1.0, 1.5],     # gamma (decay rate)
        [0.7, 0.8, 0.9]      # epsilon (asymptotic error)
    ]
    
    params, _ = try_fit_function(error_scaling, loss, error, param_ranges)
    return params

def fit_scaling_laws(dataset, val_dataset, downstream, model_dir, eval_dir, cc_mults):
    """Fit scaling laws for a given dataset configuration"""
    # Load all models for this dataset
    models = []
    for filename in os.listdir(model_dir):
        if not filename.endswith('.json'):
            continue
        model = load_model_data(f"{model_dir}/{filename}", cc_mults, [dataset], eval_dir)
        if model and f"err_{downstream}" in model:
            models.append(model)
    
    if not models:
        print(f"No valid models found for {dataset} -> {downstream}")
        return None, None, None
    
    df = pd.DataFrame(models)
    
    try:
        # Extract data for fitting
        N = df['N'].values
        M = df['tok_mult'].values
        loss = df[f'loss_{val_dataset}'].values
        error = df[f'err_{downstream}'].values
        
        # Fit both scaling laws
        loss_params, loss_type = fit_loss_scaling(N, M, loss)
        error_params = fit_error_scaling(loss, error)
        
        return loss_params, error_params, loss_type
    except Exception as e:
        print(f"Error fitting scaling laws for {dataset} -> {downstream}: {str(e)}")
        return None, None, None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "exp_data/models")
    eval_dir = os.path.join(base_dir, "exp_data/evals")
    
    datasets = ["c4_original", "rpj", "rw_original"]
    cc_mults = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    val_dataset = "c4_val"
    
    for dataset in datasets:
        for downstream in ["avg", "avg_subset"]:
            loss_params, error_params, loss_type = fit_scaling_laws(
                dataset, val_dataset, downstream, 
                model_dir, eval_dir, cc_mults
            )
            if loss_params is not None and error_params is not None:
                print(f"\nScaling laws for {dataset} -> {downstream}:")
                print(f"Loss scaling ({loss_type}): {loss_params}")
                print(f"Error scaling: {error_params}")

if __name__ == "__main__":
    main()
