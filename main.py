import math

def upstream_loss(tokens: float, flops: float) -> float:
    """Compute upstream loss (Hoffmann et al., 2022: arxiv.org/abs/2203.15556)"""
    N = flops / (6 * tokens)
    return 1.69 + 406.4 / (N**0.34) + 410.7 / (tokens**0.28)

def downstream_error(loss: float) -> float:
    """Compute downstream error (Gadre et al., 2024: arxiv.org/abs/2403.08540v2)"""
    return 0.857 - 2.21 * math.exp(-0.715 * loss)


# define scenarios
scenarios = [
    ("gpt-2", 4e10, 15e19), # double check
    ("gpt-3", 3e11, 3e23), # double check
    ("llama-400b", 15e12, 4e25),
    ("$10B cluster, 150T", 15e13, 4e26),
    ("$100B cluster, 150T", 15e13, 4e27),
    ("$100T cluster, 15000T", 15e15, 4e31),
]

for name, tokens, flops in scenarios:
    loss = upstream_loss(tokens,flops)
    error = downstream_error(loss)
    tuple = (name, tokens, flops, loss, error)
    print("{:<25} {:.2e} {:.2e} {:.4f} {:.4f}".format(*tuple))


# results
#
# gpt-2                     4.00e+10 1.50e+20 2.5468 0.4993
# gpt-3                     3.00e+11 3.00e+23 2.0033 0.3294
# llama-400b                1.50e+13 4.00e+25 1.8185 0.2549
# $10B cluster, 150T        1.50e+14 4.00e+26 1.7786 0.2374
# $100B cluster, 150T       1.50e+14 4.00e+27 1.7544 0.2266
# $100T cluster, 15000T     1.50e+16 4.00e+31 1.7064 0.2046
