import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from main import upstream_loss, downstream_error, optimized_scenarios

# Define color scheme
blue_color = '#1f77b4'
red_color = '#d62728'

# Extract token (C) and flop (D) values from optimized_scenarios
tokens_scenarios = [scenario[1] for scenario in optimized_scenarios]
flops_scenarios = [scenario[3] for scenario in optimized_scenarios]  # Use D_opt instead of D_init

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
    np.log10(tokens_grid), np.log10(flops_grid), loss_values_masked, cmap='viridis', alpha=0.3, edgecolor='none'
)

# Plot explicit data points for upstream loss
colors = [blue_color, blue_color, blue_color, blue_color, red_color, red_color, red_color]
for (label, C, D_init, D_opt, _, _), color in zip(optimized_scenarios, colors):
    loss = upstream_loss(C, D_opt)
    if loss <= 3:
        ax1.scatter(np.log10(C), np.log10(D_opt), loss, c=color, s=80, label=label, marker='x', edgecolors='w', linewidth=1.5)
        ax1.plot([np.log10(C), np.log10(C)], [np.log10(D_opt), np.log10(D_opt)], [np.nanmin(loss_values_masked), loss],
                 color=color, linestyle='--', linewidth=1)
        # Significantly shift the label to the upper left of the data point
        ax1.text(np.log10(C), np.log10(D_opt), loss + 0.07, f'{loss:.2f}', fontsize=9, color=color, ha='center')

ax1.set_xlabel('Log10 Tokens (C)', fontsize=12)
ax1.set_ylabel('Log10 Flops (D)', fontsize=12)
ax1.set_zlabel('Upstream Loss', fontsize=12)
ax1.set_title('Upstream Loss', fontsize=14, weight='bold')
ax1.view_init(elev=20, azim=60)  # Adjusted viewing angle for better data point visibility
ax1.invert_xaxis()  # Invert x-axis to reverse token scale
ax1.grid(True)

# Plot for downstream error cropped to 0.6 and below
surf2 = ax2.plot_surface(
    np.log10(tokens_grid), np.log10(flops_grid), downstream_error_values_cropped, cmap='plasma', alpha=0.3, edgecolor='none'
)

# Plot explicit data points for downstream error
for (label, C, D_init, D_opt, _, _), color in zip(optimized_scenarios, colors):
    loss = upstream_loss(C, D_opt)
    error = downstream_error(loss)
    if error <= 0.6:  # Adjusted limit to show only errors below 0.6
        ax2.scatter(np.log10(C), np.log10(D_opt), error, c=color, s=80, label=label, marker='x', edgecolors='w', linewidth=1.5)
        ax2.plot([np.log10(C), np.log10(C)], [np.log10(D_opt), np.log10(D_opt)], [np.nanmin(downstream_error_values_cropped), error],
                 color=color, linestyle='--', linewidth=1)
        # Significantly shift the label to the upper left of the data point
        ax2.text(np.log10(C), np.log10(D_opt), error + 0.015, f'{error:.2f}', fontsize=9, color=color, ha='center')

ax2.set_xlabel('Log10 Tokens (C)', fontsize=12)
ax2.set_ylabel('Log10 Flops (D)', fontsize=12)
ax2.set_zlabel('Downstream Error', fontsize=12)
ax2.set_title('Downstream Error', fontsize=14, weight='bold')
ax2.view_init(elev=20, azim=60)  # Adjusted viewing angle for better data point visibility
ax2.invert_xaxis()  # Invert x-axis to reverse token scale
ax2.grid(True)

# Create a main title for the entire figure
fig.suptitle('LLM Scaling', fontsize=16, weight='bold')

# Create a single legend for the entire figure, positioned lower to avoid overlap
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=10, ncol=1, title='Models')

# Adjust layout to make room for the legend and ensure padding
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Add spacing between the two plots
plt.subplots_adjust(wspace=0.15)

# Display the plot
plt.show()
