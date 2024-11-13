import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import pickle
from pathlib import Path
import json
import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Optional

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
    """Fit loss scaling with power law model only"""
    def power_scaling(x, alpha, beta, b, E):
        N, M = x
        return alpha * np.power(N, b) + beta * np.power(M * N, b) + E
    
    power_ranges = [
        [1e2, 1e3, 1e4],     # alpha
        [1e2, 1e3, 1e4],     # beta
        [-0.3, -0.2, -0.1],  # b
        [0.0, 1e2, 1e3]      # E
    ]
    
    power_params, power_residual = try_fit_function(
        power_scaling, (N, M), loss, power_ranges
    )
    
    return power_params

def fit_error_scaling(loss, error):
    """Fit error scaling with sigmoid and exponential decay models"""
    def sigmoid_scaling(L, k, b, c, d):
        return d - k / (1 + np.exp(-b * (L - c)))
        
    def exp_scaling(L, k, gamma, epsilon):
        return epsilon - k * np.exp(-gamma * L)
    
    # Try both sigmoid and exponential fits
    sigmoid_ranges = [
        [0.1, 0.2, 0.3],     # k (amplitude)
        [1.0, 2.0, 3.0],     # b (steepness)
        [2.0, 3.0, 4.0],     # c (midpoint)
        [0.7, 0.8, 0.9]      # d (max value)
    ]
    
    exp_ranges = [
        [0.1, 0.2, 0.3],     # k (amplitude)
        [0.5, 1.0, 1.5],     # gamma (decay rate)
        [0.7, 0.8, 0.9]      # epsilon (asymptotic error)
    ]
    
    sigmoid_params, sigmoid_residual = try_fit_function(
        sigmoid_scaling, loss, error, sigmoid_ranges
    )
    
    exp_params, exp_residual = try_fit_function(
        exp_scaling, loss, error, exp_ranges
    )
    
    return sigmoid_params, exp_params

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
        loss_params = fit_loss_scaling(N, M, loss)
        sigmoid_params, exp_params = fit_error_scaling(loss, error)
        
        print(f"\nScaling laws for {dataset} -> {downstream}:")
        print(f"Loss scaling: {loss_params}")
        print(f"Error scaling (sigmoid): {sigmoid_params}")
        print(f"Error scaling (exponential): {exp_params}")
        
        return loss_params, sigmoid_params, exp_params
    except Exception as e:
        print(f"Error fitting scaling laws for {dataset} -> {downstream}: {str(e)}")
        return None, None, None

def plot_scaling_laws(
    C_values: List[float],
    D_values: List[float],
    loss_params: Tuple,
    exp_params: Tuple,
    scenario_points: Optional[List[Tuple[str, float, float]]] = None,
    save_path: str = "scaling_laws.png",
    error_model: str = "exponential"
) -> None:
    """Plot both upstream loss and downstream error scaling laws
    
    Args:
        C_values: List of compute values
        D_values: List of token values
        loss_params: Parameters (alpha, beta, b, E) for loss scaling
        exp_params: Parameters for error scaling (exponential or sigmoid)
        scenario_points: Optional list of (name, C, D) tuples to plot
        save_path: Where to save the plot
        error_model: Which error model to use - "sigmoid" or "exponential"
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate N and M values
    N_values = [C / (6 * D) for C, D in zip(C_values, D_values)]
    M_values = [(6 * D**2) / C for C, D in zip(C_values, D_values)]
    
    # Calculate losses and errors
    losses = [
        loss_params[0] * N**loss_params[2] + 
        loss_params[1] * (M * N)**loss_params[2] + 
        loss_params[3]
        for N, M in zip(N_values, M_values)
    ]
    
    errors = [
        downstream_error(loss, exp_params, model=error_model)
        for loss in losses
    ]
    
    # Plot 1: Loss vs Compute
    ax1.scatter(C_values, losses, alpha=0.5, label='Predicted Loss')
    if scenario_points:
        scenario_C = [p[1] for p in scenario_points]
        scenario_losses = [upstream_loss(p[1], p[2], loss_params) for p in scenario_points]
        ax1.scatter(scenario_C, scenario_losses, marker='*', s=100, label='Scenarios')
        for i, (name, _, _) in enumerate(scenario_points):
            ax1.annotate(name, (scenario_C[i], scenario_losses[i]))
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Compute (C)')
    ax1.set_ylabel('Upstream Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Error vs Loss
    ax2.scatter(losses, errors, alpha=0.5, label='Predicted Error')
    if scenario_points:
        scenario_losses = [upstream_loss(p[1], p[2], loss_params) for p in scenario_points]
        scenario_errors = [downstream_error(loss, exp_params, model=error_model) for loss in scenario_losses]
        ax2.scatter(scenario_losses, scenario_errors, marker='*', s=100, label='Scenarios')
        for i, (name, _, _) in enumerate(scenario_points):
            ax2.annotate(name, (scenario_losses[i], scenario_errors[i]))
    
    ax2.set_xlabel('Upstream Loss')
    ax2.set_ylabel('Downstream Error')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_scaling_params():
    """Load the original fitted parameters from scaling_law_dict.pkl"""
    with open('scaling_law_dict.pkl', 'rb') as f:
        return pickle.load(f)

def analyze_scenarios(
    scenarios: List[Tuple[str, float, float]],
    loss_params: Optional[Tuple] = None,
    error_params: Optional[Tuple] = None,
    dataset: str = "c4_original",
    error_model: str = "exponential",  # or "sigmoid"
) -> None:
    if loss_params is None:
        # Load the exact parameters from the original paper
        scaling_dict = load_scaling_params()
        key = f"train={dataset}-loss=c4_val-downstream=avg"
        loss_params = scaling_dict[key]["loss_scaling"]
            
    if error_params is None:
        # Default parameters for error scaling
        if error_model == "exponential":
            error_params = (1.2885, 0.9722, 0.7589)  # k, gamma, epsilon (from original paper for c4_original)
        else:  # sigmoid
            error_params = (0.2, 2.0, 3.0, 0.8)  # k, b, c, d
    
    # Generate points for plotting
    C_range = np.logspace(20, 29, 100)
    D_range = np.logspace(9, 16, 100)
    
    # Create plots
    plot_scaling_laws(
        C_range.tolist(),
        D_range.tolist(),
        loss_params,
        error_params,
        scenarios,
        f"scaling_analysis_{dataset}_{error_model}.png",
        error_model=error_model
    )
    
    # Print analysis
    print(f"\nScenario Analysis using {dataset} dataset and {error_model} error scaling:")
    print(f"{'Name':<35} {'C':>12} {'D':>12} {'Loss':>10} {'Error':>10}")
    print("-" * 80)
    
    for name, C, D in scenarios:
        loss = upstream_loss(C, D, loss_params)
        error = downstream_error(loss, error_params, model=error_model)
        print(f"{name:<35} {C:>12.2e} {D:>12.2e} {loss:>10.4f} {error:>10.4f}")

def upstream_loss(C: float, D: float, loss_params: Tuple) -> float:
    """Calculate upstream loss given compute C and tokens D"""
    N = C / (6 * D)  # Parameters
    M = (6 * D**2) / C  # Tokens per parameter
    
    alpha, beta, b, E = loss_params
    return alpha * N**b + beta * (M * N)**b + E

def downstream_error(loss: float, error_params: Tuple, model: str = "exponential") -> float:
    """Calculate downstream error given upstream loss
    
    Args:
        loss: Upstream loss value
        error_params: Model parameters
        model: Which error model to use - "sigmoid" or "exponential"
    """
    if model == "exponential":
        k, gamma, epsilon = error_params
        return epsilon - k * math.exp(-gamma * loss)
    else:  # sigmoid
        k, b, c, d = error_params
        return d - k / (1 + np.exp(-b * (loss - c)))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "gadre/exp_data/models")
    eval_dir = os.path.join(base_dir, "gadre/exp_data/evals")
    
    datasets = ["c4_original", "rpj", "rw_original"]
    cc_mults = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    val_dataset = "c4_val"
    
    for dataset in datasets:
        for downstream in ["avg", "avg_subset"]:
            # Just let fit_scaling_laws handle the printing
            fit_scaling_laws(
                dataset, val_dataset, downstream, 
                model_dir, eval_dir, cc_mults
            )
    
    # Example usage with different scaling laws:
    scenarios = [
        ("GPT-2", 1e21, 2.1e10),
        ("GPT-3", 3.14e23, 3e11),
        ("GPT-4", 2e25, 1.3e13),
        ("GPT-5P", 8e26, 1.5e13),
        ("GPT-6P", 8e27, 1.5e14),
        ("GPT-7P", 8e28, 1.5e15),
    ]
    
    # Get fitted parameters for different datasets
    for dataset in ["c4_original", "rpj", "rw_original"]:
        loss_params, sigmoid_params, exp_params = fit_scaling_laws(
            dataset, "c4_val", "avg", model_dir, eval_dir, cc_mults
        )
        
        if loss_params is not None:
            # Analyze with exponential error scaling
            analyze_scenarios(scenarios, loss_params, exp_params, 
                            dataset=dataset, error_model="exponential")
            
            # Analyze with sigmoid error scaling
            analyze_scenarios(scenarios, loss_params, sigmoid_params, 
                            dataset=dataset, error_model="sigmoid")

if __name__ == "__main__":
    main()
