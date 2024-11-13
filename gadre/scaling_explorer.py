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
                
                # Calculate full average (avg)
                model["err_avg"] = np.mean([1.0 - v for v in metrics.values()])
                
                # Calculate subset average (avg_subset)
                subset_metrics = [1.0 - metrics[k] for k in SUBSET_TASKS if k in metrics]
                if subset_metrics:
                    model["err_avg_subset"] = np.mean(subset_metrics)
                
    return model

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
        return None, None
    
    df = pd.DataFrame(models)
    
    try:
        # Extract data for fitting
        N = df['N'].values
        M = df['tok_mult'].values
        loss = df[f'loss_{val_dataset}'].values
        error = df[f'err_{downstream}'].values
        
        # Fit loss scaling law: L(N,M) = α/(1+exp(-bN)) + β/(1+exp(-bMN)) + E
        def loss_scaling(x, alpha, beta, b, E):
            N, M = x
            return (alpha / (1 + np.exp(-b * N))) + (beta / (1 + np.exp(-b * M * N))) + E
        
        # Fit error scaling law: E(L) = ε - k * exp(-γL)
        def error_scaling(L, k, gamma, epsilon):
            return epsilon - k * np.exp(-gamma * L)
        
        loss_params, _ = curve_fit(loss_scaling, (N, M), loss, maxfev=10000)
        error_params, _ = curve_fit(error_scaling, loss, error, maxfev=10000)
        
        return loss_params, error_params
    except Exception as e:
        print(f"Error fitting scaling laws for {dataset} -> {downstream}: {str(e)}")
        return None, None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "exp_data/models")
    eval_dir = os.path.join(base_dir, "exp_data/evals")
    
    datasets = ["c4_original", "rpj", "rw_original"]
    cc_mults = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    val_dataset = "c4_val"
    
    for dataset in datasets:
        for downstream in ["avg", "avg_subset"]:
            loss_params, error_params = fit_scaling_laws(
                dataset, val_dataset, downstream, 
                model_dir, eval_dir, cc_mults
            )
            if loss_params is not None and error_params is not None:
                print(f"\nScaling laws for {dataset} -> {downstream}:")
                print(f"Loss scaling (α, β, b, E): {loss_params}")
                print(f"Error scaling (k, γ, ε): {error_params}")

if __name__ == "__main__":
    main()