import math

def upstream_loss(D: float, C: float, E: float = 1.84, alpha: float = 212, beta: float = 367, eta: float = 0.136) -> float:
    """Compute upstream loss (Gadre et al., 2024: arxiv.org/pdf/2403.08540)"""
    # N = C / (6 * D)
    # M = D / N
    M = (6*D**2)/C
    return E + (alpha * (M**eta) + beta * (M**-eta)) * C**-eta

def downstream_error(loss: float, epsilon: float = 0.857, k: float = 2.21, gamma: float = 0.715) -> float:
    """Compute downstream error (Gadre et al., 2024: arxiv.org/abs/2403.08540v2)"""
    return epsilon - k * math.exp(-gamma * loss)


# define scenarios (names, D, C)
scenarios = [
    ("GPT-2", 4e10, 1e20),
    ("GPT-3", 3e11, 1e23),
    ("LLaMA-400B", 1.4e12, 3e24),
    ("GPT-4", 1e13, 6e25),
    ("[predict] $10B (1M GPUs), 150T tokens", 1.5e14, 6e26),
    ("[predict] $100B (10M GPUs), 150T tokens", 1.5e14, 6e27),
    ("[predict] $1T (100M GPUs), 150T tokens", 1.5e14, 6e28),
]

for name, D, C in scenarios:
    loss = upstream_loss(D, C)
    error = downstream_error(loss)
    print(f"{name:<25} {D:.2e} {C:.2e} {loss:.4f} {error:.4f}")
