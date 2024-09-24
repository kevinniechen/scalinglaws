import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from main import upstream_loss, downstream_error, scenarios

# Define color scheme
blue_color = '#1f77b4'
red_color = '#d62728'

# Extract token (C) and flop (D) values from scenarios
tokens_scenarios = [scenario[1] for scenario in scenarios]
flops_scenarios = [scenario[2] for scenario in scenarios]

# Determine the exponent range based on scenarios and extend the range
buffer = 1  # Define how much to extend the range
min_token_exp = math.floor(np.log10(min(tokens_scenarios))) - buffer
max_token_exp = math.ceil(np.log10(max(tokens_scenarios))) + buffer
min_flop_exp = math.floor(np.log10(min(flops_scenarios))) - buffer
max_flop_exp = math.ceil(np.log10(max(flops_scenarios))) + buffer

# Create the token and flops range ensuring coverage of all scenario points plus the buffer
tokens = np.logspace(min_token_exp, max_token_exp, 100)
flops = np.logspace(min_flop_exp, max_flop_exp, 100)
tokens_grid, flops_grid = np.meshgrid(tokens, flops)

# Compute upstream loss values
loss_values = upstream_loss(tokens_grid, flops_grid)

# Convert upstream loss to downstream error using the downstream_error function
downstream_error_values = np.vectorize(downstream_error)(loss_values)

# Mask the loss values to only show those below 
loss_values_masked = np.where(loss_values <= 3, loss_values, np.nan)

# Mask the downstream error values to only show those below
downstream_error_values_cropped = np.where(downstream_error_values <= 0.6, downstream_error_values, np.nan)

# Plot side-by-side graphs with significantly shifted labels to avoid overlap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})

# Plot for upstream loss with adjusted orientation
surf1 = ax1.plot_surface(
    np.log10(tokens_grid), np.log10(flops_grid), loss_values_masked, cmap='viridis', alpha=0.3
)

# Plot explicit data points for upstream loss
colors = [blue_color, blue_color, blue_color, blue_color, red_color, red_color, red_color]
for (label, C, D), color in zip(scenarios, colors):
    loss = upstream_loss(C, D)
    if loss <= 3:
        ax1.scatter(np.log10(C), np.log10(D), loss, c=color, s=80, label=label, marker='x')
        ax1.plot([np.log10(C), np.log10(C)], [np.log10(D), np.log10(D)], [np.nanmin(loss_values_masked), loss],
                 color=color, linestyle='--', linewidth=1)
        # Significantly shift the label to the upper left of the data point
        ax1.text(np.log10(C), np.log10(D), loss + 0.07, f'{loss:.2f}', fontsize=8, color=color)

ax1.set_xlabel('Log10 Tokens (C)')
ax1.set_ylabel('Log10 Flops (D)')
ax1.set_zlabel('Upstream Loss')
ax1.set_title('Upstream Loss')
ax1.view_init(elev=15, azim=45)  # Adjusted viewing angle for better data point visibility
ax1.invert_xaxis()  # Invert x-axis to reverse token scale

# Plot for downstream error cropped to 0.6 and below
surf2 = ax2.plot_surface(
    np.log10(tokens_grid), np.log10(flops_grid), downstream_error_values_cropped, cmap='plasma', alpha=0.3
)

# Plot explicit data points for downstream error
for (label, C, D), color in zip(scenarios, colors):
    loss = upstream_loss(C, D)
    error = downstream_error(loss)
    if error <= 0.6:  # Adjusted limit to show only errors below 0.6
        ax2.scatter(np.log10(C), np.log10(D), error, c=color, s=80, label=label, marker='x')
        ax2.plot([np.log10(C), np.log10(C)], [np.log10(D), np.log10(D)], [np.nanmin(downstream_error_values_cropped), error],
                 color=color, linestyle='--', linewidth=1)
        # Significantly shift the label to the upper left of the data point
        ax2.text(np.log10(C), np.log10(D), error + 0.015, f'{error:.2f}', fontsize=8, color=color)

ax2.set_xlabel('Log10 Tokens (C)')
ax2.set_ylabel('Log10 Flops (D)')
ax2.set_zlabel('Downstream Error')
ax2.set_title('Downstream Error')
ax2.view_init(elev=15, azim=45)  # Adjusted viewing angle for better data point visibility
ax2.invert_xaxis()  # Invert x-axis to reverse token scale

# Create a single legend for the entire figure, positioned lower to avoid overlap
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=10, title='Scenarios')

# Adjust layout to make room for the legend and ensure padding
plt.tight_layout(rect=[0, 0, 0.9, 1])

# Add spacing between the two plots
plt.subplots_adjust(wspace=0.1)

plt.show()

