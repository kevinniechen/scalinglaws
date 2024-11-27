import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from main import upstream_loss, downstream_error, optimized_scenarios

def create_plots(layout='vertical'):
    # Define color scheme
    blue_color = '#1f77b4'
    red_color = '#d62728'

    # Modify the scenario labels to be simpler
    modified_scenarios = []
    for scenario in optimized_scenarios:
        label = scenario[0]
        if "GPT-5P" in label:
            label = "GPT-5P (1M GPUs)"
        elif "GPT-6P" in label:
            label = "GPT-6P (10M GPUs)"
        elif "GPT-7P" in label:
            label = "GPT-7P (100M GPUs)"
        modified_scenarios.append((label,) + scenario[1:])

    # Extract token (C) and flop (D) values from modified_scenarios
    tokens_scenarios = [scenario[1] for scenario in modified_scenarios]
    flops_scenarios = [scenario[3] for scenario in modified_scenarios]

    # Determine the exponent range based on scenarios and extend the range
    buffer = 1
    min_token_exp = math.floor(np.log10(min(tokens_scenarios))) - buffer
    max_token_exp = math.ceil(np.log10(max(tokens_scenarios))) + buffer
    min_flop_exp = math.floor(np.log10(min(flops_scenarios))) - buffer
    max_flop_exp = math.ceil(np.log10(max(flops_scenarios))) + buffer

    # Create the token and flops range
    tokens = np.logspace(min_token_exp, max_token_exp, 100)
    flops = np.logspace(min_flop_exp, max_flop_exp, 100)
    tokens_grid, flops_grid = np.meshgrid(tokens, flops)

    # Compute values
    loss_values = upstream_loss(tokens_grid, flops_grid)
    downstream_error_values = np.vectorize(downstream_error)(loss_values)
    loss_values_masked = np.where(loss_values <= 3, loss_values, np.nan)
    downstream_error_values_cropped = np.where(downstream_error_values <= 0.6, downstream_error_values, np.nan)

    # Create figure with appropriate dimensions for layout
    if layout == 'vertical':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 24), subplot_kw={'projection': '3d'})
    else:  # horizontal
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), subplot_kw={'projection': '3d'})

    # Plot surfaces
    surf1 = ax1.plot_surface(np.log10(tokens_grid), np.log10(flops_grid), loss_values_masked, 
                            cmap='viridis', alpha=0.3, edgecolor='none')
    
    # Plot data points
    colors = [blue_color, blue_color, blue_color, red_color, red_color, red_color, red_color]
    for (label, C, D_init, D_opt, _, _), color in zip(modified_scenarios, colors):
        loss = upstream_loss(C, D_opt)
        if loss <= 3:
            ax1.scatter(np.log10(C), np.log10(D_opt), loss, 
                       c=color, s=160, label=label, marker='x', linewidth=4)
            ax1.plot([np.log10(C), np.log10(C)], [np.log10(D_opt), np.log10(D_opt)], 
                     [np.nanmin(loss_values_masked), loss],
                     color=color, linestyle='--', linewidth=2)
            ax1.text(np.log10(C), np.log10(D_opt), loss + 0.1, 
                    f'{loss:.2f}', fontsize=16, color=color, ha='center', weight='bold')

    ax1.set_xlabel('Log10 Tokens (C)', fontsize=24, labelpad=20)
    ax1.set_ylabel('Log10 Flops (D)', fontsize=24, labelpad=20)
    ax1.set_zlabel('Upstream Loss', fontsize=24, labelpad=20)
    ax1.set_title('Upstream Loss', fontsize=28, pad=20)
    ax1.view_init(elev=25, azim=45)
    ax1.invert_xaxis()
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # Plot for downstream error
    surf2 = ax2.plot_surface(np.log10(tokens_grid), np.log10(flops_grid), downstream_error_values_cropped, 
                            cmap='plasma', alpha=0.3, edgecolor='none')

    for (label, C, D_init, D_opt, _, _), color in zip(modified_scenarios, colors):
        loss = upstream_loss(C, D_opt)
        error = downstream_error(loss)
        if error <= 0.6:
            ax2.scatter(np.log10(C), np.log10(D_opt), error, 
                       c=color, s=160, label=label, marker='x', linewidth=4)
            ax2.plot([np.log10(C), np.log10(C)], [np.log10(D_opt), np.log10(D_opt)], 
                     [np.nanmin(downstream_error_values_cropped), error],
                     color=color, linestyle='--', linewidth=2)
            ax2.text(np.log10(C), np.log10(D_opt), error + 0.02, 
                    f'{error:.2f}', fontsize=16, color=color, ha='center', weight='bold')

    ax2.set_xlabel('Log10 Tokens (C)', fontsize=24, labelpad=20)
    ax2.set_ylabel('Log10 Flops (D)', fontsize=24, labelpad=20)
    ax2.set_zlabel('Downstream Error', fontsize=24, labelpad=20)
    ax2.set_title('Downstream Error', fontsize=28, pad=20)
    ax2.view_init(elev=25, azim=45)
    ax2.invert_xaxis()
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    # Create a single legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, 
              loc='center', 
              bbox_to_anchor=(0.5, 0.02),
              ncol=2,
              fontsize=20)

    # Adjust layout
    plt.tight_layout()
    if layout == 'vertical':
        plt.subplots_adjust(bottom=0.1, hspace=0.3)
    else:
        plt.subplots_adjust(bottom=0.1, wspace=0.2)

    # Save the figure
    filename = 'scaling_laws_3d_' + layout + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create both layouts
create_plots('vertical')
create_plots('horizontal')
