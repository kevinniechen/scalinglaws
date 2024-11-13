import os
import numpy as np
from scipy.optimize import curve_fit
import pickle
import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Optional

def load_scaling_params():
    """Load the original fitted parameters from scaling_law_dict.pkl"""
    import os
    pickle_path = os.path.join(os.path.dirname(__file__), 'gadre', 'scaling_law_dict.pkl')
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def fit_error_scaling(loss: np.ndarray, error: np.ndarray):
    """Fit error scaling with sigmoid and exponential decay models"""
    def sigmoid_scaling(L, k, b, c, d):
        # Clip to avoid overflow
        x = np.clip(-b * (L - c), -100, 100)
        return d - k / (1 + np.exp(x))
        
    def exp_scaling(L, k, gamma, epsilon):
        # Clip to avoid overflow
        x = np.clip(-gamma * L, -100, 100)
        return epsilon - k * np.exp(x)
    
    # Try both sigmoid and exponential fits
    try:
        exp_params, _ = curve_fit(
            exp_scaling, loss, error,
            p0=[1.2, 1.0, 0.75],  # Initial guesses closer to their values
            bounds=([0, 0, 0], [5, 5, 1])  # Tighter bounds based on expected values
        )
        print(f"Successfully fit exponential with parameters: k={exp_params[0]:.3f}, gamma={exp_params[1]:.3f}, epsilon={exp_params[2]:.3f}")
    except Exception as e:
        print(f"Failed to fit exponential: {e}")
        exp_params = None
        
    try:
        sigmoid_params, _ = curve_fit(
            sigmoid_scaling, loss, error,
            p0=[0.5, 1.0, 2.0, 0.8],  # Initial guesses for k, b, c, d
            bounds=([0, 0, 0, 0], [5, 5, 5, 1])  # Tighter bounds
        )
        print(f"Successfully fit sigmoid with parameters: k={sigmoid_params[0]:.3f}, b={sigmoid_params[1]:.3f}, c={sigmoid_params[2]:.3f}, d={sigmoid_params[3]:.3f}")
    except Exception as e:
        print(f"Failed to fit sigmoid: {e}")
        sigmoid_params = None
    
    return sigmoid_params, exp_params

def plot_error_scaling(loss: np.ndarray, error: np.ndarray, exp_params: Tuple, sigmoid_params: Tuple, their_exp_params: Tuple, save_path: str):
    """Plot error scaling fits against data"""
    plt.figure(figsize=(10, 6))
    
    # Plot original data points
    plt.scatter(loss, error, alpha=0.5, label='Data Points')
    
    # Generate points for smooth curves
    loss_range = np.linspace(min(loss), max(loss), 100)
    
    # Plot their exponential fit
    k, gamma, epsilon = their_exp_params
    their_exp_errors = epsilon - k * np.exp(-gamma * loss_range)
    plt.plot(loss_range, their_exp_errors, 'b-', label=f'Their Exponential (k={k:.3f}, γ={gamma:.3f}, ε={epsilon:.3f})')
    
    # Plot our exponential fit
    if exp_params is not None:
        k, gamma, epsilon = exp_params
        exp_errors = epsilon - k * np.exp(-gamma * loss_range)
        plt.plot(loss_range, exp_errors, 'r-', label=f'Our Exponential (k={k:.3f}, γ={gamma:.3f}, ε={epsilon:.3f})')
    
    # Plot our sigmoid fit
    if sigmoid_params is not None:
        k, b, c, d = sigmoid_params
        sigmoid_errors = d - k / (1 + np.exp(-b * (loss_range - c)))
        plt.plot(loss_range, sigmoid_errors, 'g-', label=f'Our Sigmoid (k={k:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f})')
    
    plt.xlabel('Upstream Loss')
    plt.ylabel('Downstream Error')
    plt.grid(True)
    plt.legend()
    plt.title('Error Scaling: Comparing Different Fits')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scenario_analysis(scenarios: List[Tuple[str, float, float]], 
                         loss_params: Tuple,
                         exp_params: Tuple,
                         sigmoid_params: Tuple,
                         their_exp_params: Tuple,
                         save_path: str = "scenario_analysis.png"):
    """Plot scenario analysis comparing different error scaling approaches"""
    
    def upstream_loss(C: float, D: float) -> float:
        """Calculate upstream loss given compute C and tokens D"""
        N = C / (6 * D)  # Parameters
        M = (6 * D**2) / C  # Tokens per parameter
        
        alpha, beta, b, E = loss_params
        return alpha * N**b + beta * (M * N)**b + E
    
    def downstream_error_exp(loss: float, params: Tuple) -> float:
        k, gamma, epsilon = params
        return epsilon - k * math.exp(-gamma * loss)
    
    def downstream_error_sigmoid(loss: float) -> float:
        k, b, c, d = sigmoid_params
        return d - k / (1 + np.exp(-b * (loss - c)))
    
    # Calculate losses and errors for each scenario
    names = [s[0] for s in scenarios]
    losses = [upstream_loss(s[1], s[2]) for s in scenarios]
    their_exp_errors = [downstream_error_exp(l, their_exp_params) for l in losses]
    our_exp_errors = [downstream_error_exp(l, exp_params) for l in losses]
    sigmoid_errors = [downstream_error_sigmoid(l) for l in losses]
    
    # Create plot
    plt.figure(figsize=(15, 6))
    x = range(len(scenarios))
    width = 0.25
    
    plt.bar([i - width for i in x], their_exp_errors, width, label='Their Exponential', color='blue', alpha=0.6)
    plt.bar([i for i in x], our_exp_errors, width, label='Our Exponential', color='red', alpha=0.6)
    plt.bar([i + width for i in x], sigmoid_errors, width, label='Our Sigmoid', color='green', alpha=0.6)
    
    plt.xlabel('Model')
    plt.ylabel('Predicted Error Rate')
    plt.title('Error Predictions: Comparing Different Scaling Laws')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 1. Load existing parameters from pickle
    scaling_dict = load_scaling_params()
    key = "train=c4_original-loss=c4_val-downstream=avg"
    
    # Get their fitted parameters
    loss_params = scaling_dict[key]["loss_scaling"]
    their_error_params = scaling_dict[key]["error_scaling"]
    
    print("\nOriginal fitted parameters:")
    print(f"Loss scaling: alpha={loss_params[0]:.3f}, beta={loss_params[1]:.3f}, b={loss_params[2]:.3f}, E={loss_params[3]:.3f}")
    print(f"Error scaling (exponential): k={their_error_params[0]:.3f}, gamma={their_error_params[1]:.3f}, epsilon={their_error_params[2]:.3f}")
    
    # Get the loss and error data points, ensuring they match
    loss_data = scaling_dict[key]["loss_points"]
    error_data = scaling_dict[key]["error_points"]
    
    # Use only the points where we have both loss and error data
    n_points = min(len(loss_data["loss"]), len(error_data["error"]))
    loss_points = np.array(loss_data["loss"][:n_points])
    error_points = np.array(error_data["error"][:n_points])
    
    print(f"\nFitting to {len(loss_points)} data points...")
    print(f"Loss range: [{min(loss_points):.3f}, {max(loss_points):.3f}]")
    print(f"Error range: [{min(error_points):.3f}, {max(error_points):.3f}]")
    
    print("\nData points used for fitting:")
    for i, (l, e) in enumerate(zip(loss_points, error_points)):
        print(f"Point {i}: loss={l:.3f}, error={e:.3f}")
    
    # 2. Fit our own error scaling models
    print("\nFitting our own error scaling models...")
    sigmoid_params, exp_params = fit_error_scaling(loss_points, error_points)
    
    # Plot the fits
    plot_error_scaling(
        loss_points, 
        error_points,
        exp_params,
        sigmoid_params,
        their_error_params,
        "error_scaling_fits.png"
    )
    
    # 3. Run scenario analysis
    scenarios = [
        ("GPT-2", 1e21, 2.1e10),
        ("GPT-3", 3.14e23, 3e11),
        ("GPT-4", 2e25, 1.3e13),
        ("GPT-5P", 8e26, 1.5e13),
        ("GPT-6P", 8e27, 1.5e14),
        ("GPT-7P", 8e28, 1.5e15),
    ]
    
    if exp_params is not None:
        # Plot scenario analysis
        plot_scenario_analysis(
            scenarios,
            loss_params,
            exp_params,
            sigmoid_params,
            their_error_params,
            "scenario_comparison.png"
        )
    else:
        print("\nSkipping scenario analysis because error scaling fits failed.")

if __name__ == "__main__":
    main()
