import math

def upstream_loss(C: float, D: float, E: float = 1.84, alpha: float = 212, beta: float = 367, eta: float = 0.136) -> float:
    """Compute upstream loss (Gadre et al., 2024: arxiv.org/pdf/2403.08540)"""
    # N = C / (6 * D)
    # M = D / N
    M = (6 * D**2) / C
    return E + (alpha * (M**eta) + beta * (M**-eta)) * C**-eta

def downstream_error(loss: float, epsilon: float = 0.857, k: float = 2.21, gamma: float = 0.715) -> float:
    """Compute downstream error (Gadre et al., 2024: arxiv.org/abs/2403.08540v2)"""
    return epsilon - k * math.exp(-gamma * loss)


# define scenarios (names, C, D)
scenarios = [
    ("GPT-2", 1e20, 4e10),
    ("GPT-3", 1e23, 3e11),
    ("LLaMA-400B", 3e24, 1.4e12),
    ("GPT-4", 6e25, 1e13),
    ("GPT-5P (1M ($10B) GPUs, 15T tokens)", 6e26, 1.5e13),
    ("GPT-6P (10M ($100B) GPUs, 15T tokens)", 6e27, 1.5e13),
    ("GPT-7P (100M ($1T) GPUs, 150T tokens)", 6e28, 1.5e14),
]

for name, C, D in scenarios:
    loss = upstream_loss(C, D)
    error = downstream_error(loss)
    print(f"{name:<25} {C:.2e} {D:.2e} {loss:.4f} {error:.4f}")
