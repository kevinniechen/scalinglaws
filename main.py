import math
import numpy as np
import matplotlib.pyplot as plt


def upstream_loss(
    C: float,
    D: float,
    E: float = 1.84,
    alpha: float = 212,
    beta: float = 367,
    eta: float = 0.136,
) -> float:
    """Compute upstream loss fitted on RedPajama data (Gadre et al., 2024: arxiv.org/pdf/2403.08540)"""
    M = (6 * D**2) / C  # N = C / (6 * D) and M = D / N
    return E + (alpha * (M**eta) + beta * (M**-eta)) * C**-eta


def downstream_error(
    loss: float, epsilon: float = 0.857, k: float = 2.21, gamma: float = 0.715
) -> float:
    """Compute downstream error fitted on RedPajama data (Gadre et al., 2024: arxiv.org/abs/2403.08540v2)"""
    return epsilon - k * math.exp(-gamma * loss)


def gradient_descent_multiplicative(
    C: float, D_init: float, factor: float = 1.1, num_iterations: int = 1000
) -> float:
    """Perform gradient descent with multiplicative steps to find optimal D for a given C"""
    D = D_init
    initial_loss = upstream_loss(C, D)
    print(f"Initial D: {D:.2e}, Initial loss: {initial_loss:.6f}")

    for i in range(num_iterations):
        current_loss = upstream_loss(C, D)
        loss_up = upstream_loss(C, D * factor)
        loss_down = upstream_loss(C, D / factor)

        if loss_up < current_loss and loss_up < loss_down:
            D *= factor
        elif loss_down < current_loss and loss_down < loss_up:
            D /= factor
        else:
            # If we can't improve, we're at a local optimum
            break

        if (
            i % 10 == 0 or i == num_iterations - 1
        ):  # Print every 10 iterations and the last one
            print(f"Iteration {i}: D = {D:.2e}, Loss = {upstream_loss(C, D):.6f}")

    final_loss = upstream_loss(C, D)
    print(f"Final D: {D:.2e}, Final loss: {final_loss:.6f}")
    print(f"Total loss change: {initial_loss - final_loss:.6f}")
    return D


def format_tokens(tokens):
    """Format token count in a readable way"""
    if tokens >= 1e12:
        return f"{tokens/1e12:.1f}T"
    elif tokens >= 1e9:
        return f"{tokens/1e9:.1f}B"
    else:
        return f"{tokens/1e6:.1f}M"


# define scenarios (names, C, D)
scenarios = [
    ("GPT-2", 1e21, 2.1e10),
    ("GPT-3", 3.14e23, 3e11),
    ("GPT-4", 2e25, 1.3e13),
    # ("LLaMA-400B", 4e25, 1.5e13),
    ("GPT-5P (1M or $10B GPUs)", 8e26, 1.5e13),
    ("GPT-6P (10M or $100B GPUs)", 8e27, 1.5e14),
    ("GPT-7P (100M or $1T GPUs)", 8e28, 1.5e15),
]

optimized_scenarios = []
print(f"{'Name':<35} {'C':>10} {'D_init':>10} {'D_opt':>10} {'Loss':>10} {'Error':>10}")
print("-" * 90)

for name, C, D_init in scenarios:
    if (
        name.startswith("GPT-5P")
        or name.startswith("GPT-6P")
        or name.startswith("GPT-7P")
    ):
        print(f"\nOptimizing for {name}:")
        D_opt = gradient_descent_multiplicative(C, D_init)
    else:
        D_opt = D_init
    loss = upstream_loss(C, D_opt)
    error = downstream_error(loss)
    updated_name = f"{name} ({format_tokens(D_opt)} tokens)"
    optimized_scenarios.append((updated_name, C, D_init, D_opt, loss, error))
    print(f"{updated_name:<35} {C:.2e} {D_init:.2e} {D_opt:.2e} {loss:.4f} {error:.4f}")

# Plotting
plt.figure(figsize=(12, 8))
for name, C, D_init, D_opt, loss, error in optimized_scenarios:
    plt.scatter(C, D_opt, label=name)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Compute (C)")
plt.ylabel("Optimal Tokens (D)")
plt.title("Optimal Tokens vs Compute for Different Scenarios")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.grid(True)
plt.savefig("optimal_tokens_vs_compute.png", dpi=300, bbox_inches="tight")
plt.close()

# Save data to a file
with open("optimized_scenarios.txt", "w") as f:
    f.write(
        f"{'Name':<35} {'C':>10} {'D_init':>10} {'D_opt':>10} {'Loss':>10} {'Error':>10}\n"
    )
    f.write("-" * 90 + "\n")
    for name, C, D_init, D_opt, loss, error in optimized_scenarios:
        f.write(f"{name:<35} {C:.2e} {D_init:.2e} {D_opt:.2e} {loss:.4f} {error:.4f}\n")

print("\nOptimized scenarios data saved to 'optimized_scenarios.txt'")
print("Plot saved as 'optimal_tokens_vs_compute.png'")
