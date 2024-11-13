import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl

# Add base directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gadre.scaling.constants import *
from gadre.scaling.laws import *
from gadre.scaling.shared import *

def load_data():
    """Load and parse model data"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "exp_data/models")
    cc_mults = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    datasets = ["c4_original", "rpj", "rw_original"]
    downstreams = ["avg", "avg_subset"]
    eval_dir = os.path.join(base_dir, "exp_data/evals")
    val_dataset = "c4_val"
    
    df = parse_model_jsons(model_dir, cc_mults=cc_mults, datasets=datasets, eval_dir=eval_dir)
    return df, datasets, downstreams, val_dataset, model_dir, eval_dir, cc_mults

def fit_scaling_laws(datasets, downstreams, val_dataset, model_dir, eval_dir, cc_mults):
    """Fit scaling laws with defaults from Table 2"""
    scaling_law_dict = {}
    
    for dataset in datasets:
        for downstream in ["avg", "avg_subset"]:
            # Use 1.4B, M=20 run for fitting the top-1 error scaling laws
            ((a, b, alpha_c, E), (k, gamma, epsilon)), (loss_points, error_points) = fit_ds(
                dataset,
                val_dataset,
                downstream,
                model_dir,
                eval_dir,
                cc_mults,
                True,
                True,
            )
            
            key = f"train={dataset}-loss={val_dataset}-downstream={downstream}"
            scaling_law_dict[key] = {
                "loss_scaling": (a, b, alpha_c, E),
                "error_scaling": (k, gamma, epsilon),
                "loss_points": loss_points,
                "error_points": error_points,
            }
            
    return scaling_law_dict

def main():
    # Load data
    df, datasets, downstreams, val_dataset, model_dir, eval_dir, cc_mults = load_data()
    print(f"num models: {len(df.index)}")
    print(f"fields: {df.columns.tolist()}")
    
    # Fit scaling laws
    scaling_law_dict = fit_scaling_laws(datasets, downstreams, val_dataset, 
                                      model_dir, eval_dir, cc_mults)
    
    # Example of accessing results
    for key in scaling_law_dict:
        print(f"\nScaling laws for {key}:")
        print(f"Loss scaling parameters: {scaling_law_dict[key]['loss_scaling']}")
        print(f"Error scaling parameters: {scaling_law_dict[key]['error_scaling']}")

if __name__ == "__main__":
    main()
