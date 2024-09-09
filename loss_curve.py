import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the upstream_loss function
def upstream_loss(tokens: float, flops: float) -> float:
    """Compute upstream loss (Hoffmann et al., 2022: arxiv.org/abs/2203.15556)"""
    N = flops / (6 * tokens)
    return 1.69 + 406.4 / (N**0.34) + 410.7 / (tokens**0.28)

# Data points (tokens, flops) and labels
data_points = [
    ("gpt-2", 4e10, 15e19),
    ("gpt-3", 3e11, 3e23),
    ("llama-400b", 15e12, 4e25),
    ("$10B cluster, 150T", 15e13, 4e26),
    ("$100B cluster, 150T", 15e13, 4e27),
    ("$100T cluster, 15000T", 15e15, 4e31),
]

# Define new labels with "prediction" removed from the first three
new_labels_final = [
    "gpt-2",
    "gpt-3",
    "llama-400b",
    "prediction $10B gpus, 150T",
    "prediction $100B gpus, 150T",
    "prediction $100T gpus, 15P"  # P=quadrillion for 15000T
]

# Define color scheme
blue_color = '#1f77b4'
red_color = '#d62728'
final_colors = [blue_color, blue_color, blue_color, red_color, red_color, red_color]

# Create the token and flops range
tokens = np.logspace(np.log10(4e9), 18, 100)
flops = np.logspace(np.log10(15e18), 36, 100)
tokens_grid, flops_grid = np.meshgrid(tokens, flops)
loss_values = upstream_loss(tokens_grid, flops_grid)

# Mask the loss values to only show those below 3
loss_values_masked = np.where(loss_values <= 3, loss_values, np.nan)
min_loss_cutoff = np.nanmin(loss_values_masked)

# Prepare figure and axis again for the plot with X's, adjusted colors, and a legend
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with a color gradient, but only show loss <= 3
surf = ax.plot_surface(
    np.log10(tokens_grid), np.log10(flops_grid), loss_values_masked, cmap='viridis', alpha=0.3
)

# Plot explicit data points with X's and the final color scheme, adding labels for the legend
for (label, token, flop), new_label, color in zip(data_points, new_labels_final, final_colors):
    loss = upstream_loss(token, flop)
    if loss <= 3:
        ax.scatter(np.log10(token), np.log10(flop), loss, c=color, s=80, label=new_label, marker='x')
        # Add vertical lines from the minimum loss cutoff
        ax.plot([np.log10(token), np.log10(token)], [np.log10(flop), np.log10(flop)], [min_loss_cutoff, loss], 
                color=color, linestyle='--', linewidth=1)

# Customize the axes
ax.set_xlabel('Log10 Tokens')
ax.set_ylabel('Log10 Flops')
ax.set_zlabel('Upstream Loss')

# Adjust the viewing angle for better loss visibility
ax.view_init(elev=30, azim=120)

# Set plot limits to include the new range
ax.set_xlim(np.log10(4e9), 18)
ax.set_ylim(np.log10(15e18), 36)

# Add a legend to the plot
ax.legend(loc='upper left', fontsize=10)

# Set title
ax.set_title('Loss against tokens and flops, with known cluster buildouts')

plt.show()
