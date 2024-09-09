import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from main import upstream_loss, downstream_error , scenarios

# Define color scheme
blue_color = '#1f77b4'
red_color = '#d62728'
final_colors = [blue_color, blue_color, blue_color, red_color, red_color, red_color]

# Create the token and flops range
tokens = np.logspace(np.log10(4e9), 18, 100)
flops = np.logspace(np.log10(15e18), 36, 100)
tokens_grid, flops_grid = np.meshgrid(tokens, flops)

# Compute upstream loss values
loss_values = upstream_loss(tokens_grid, flops_grid)

# Convert upstream loss to downstream error using the downstream_error function
downstream_error_values = np.vectorize(downstream_error)(loss_values)

# Mask the loss values to only show those below 3
loss_values_masked = np.where(loss_values <= 3, loss_values, np.nan)

# Mask the downstream error values to only show those below 0.55
downstream_error_values_cropped = np.where(downstream_error_values <= 0.55, downstream_error_values, np.nan)

# Plot side-by-side graphs with significantly shifted labels to avoid overlap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})

# Plot for upstream loss with adjusted orientation
surf1 = ax1.plot_surface(
    np.log10(tokens_grid), np.log10(flops_grid), loss_values_masked, cmap='viridis', alpha=0.3
)

# Plot explicit data points for upstream loss
for (label, token, flop), color in zip(scenarios, final_colors):
    loss = upstream_loss(token, flop)
    if loss <= 3:
        ax1.scatter(np.log10(token), np.log10(flop), loss, c=color, s=80, label=label, marker='x')
        ax1.plot([np.log10(token), np.log10(token)], [np.log10(flop), np.log10(flop)], [np.nanmin(loss_values_masked), loss],
                 color=color, linestyle='--', linewidth=1)
        # Significantly shift the label to the upper left of the data point
        ax1.text(np.log10(token) - 0.3, np.log10(flop) + 0.3, loss, f'{loss:.2f}', fontsize=8, color=color)

ax1.set_xlabel('Log10 Tokens')
ax1.set_ylabel('Log10 Flops')
ax1.set_zlabel('Upstream Loss')
ax1.set_title('Upstream Loss')
ax1.legend(loc='upper left', fontsize=10)
ax1.view_init(elev=40, azim=130)  # Adjusted viewing angle for better data point visibility

# Plot for downstream error cropped to 0.55 and below
surf2 = ax2.plot_surface(
    np.log10(tokens_grid), np.log10(flops_grid), downstream_error_values_cropped, cmap='plasma', alpha=0.3
)

# Plot explicit data points for downstream error
for (label, token, flop), color in zip(scenarios, final_colors):
    loss = upstream_loss(token, flop)
    error = downstream_error(loss)
    if error <= 0.55:  # Adjusted limit to show only errors below 0.55
        ax2.scatter(np.log10(token), np.log10(flop), error, c=color, s=80, label=label, marker='x')
        ax2.plot([np.log10(token), np.log10(token)], [np.log10(flop), np.log10(flop)], [np.nanmin(downstream_error_values_cropped), error],
                 color=color, linestyle='--', linewidth=1)
        # Significantly shift the label to the upper left of the data point
        ax2.text(np.log10(token) - 0.3, np.log10(flop) + 0.3, error, f'{error:.2f}', fontsize=8, color=color)

ax2.set_xlabel('Log10 Tokens')
ax2.set_ylabel('Log10 Flops')
ax2.set_zlabel('Downstream Error')
ax2.set_title('Downstream Error')
ax2.legend(loc='upper left', fontsize=10)
ax2.view_init(elev=40, azim=130)  # Adjusted viewing angle for better data point visibility

plt.show()
