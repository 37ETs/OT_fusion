import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_distributions(classification_head_before, classification_head_after, ds, save_dir):
    """
    Compare weight distributions and heatmaps for classification_head_before and classification_head_after,
    then plot and save the figures to the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for (name_b, param_b), (name_a, param_a) in zip(
            classification_head_before.items(),
            classification_head_after.items(),
        ):
            if 'weight' in name_b and param_b.dim() == 2:
                # Distribution KDE plot
                plt.figure(figsize=(10, 4))
                sns.kdeplot(param_b.flatten().cpu().numpy(), label='Before', color='blue', alpha=0.5)
                sns.kdeplot(param_a.flatten().cpu().numpy(), label='After', color='red', alpha=0.5)
                plt.xlabel("Weight Value")
                plt.ylabel("Density")
                plt.title(f"Weights Distribution ({ds} - {name_b})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{ds}_{name_b.replace('.', '_')}_distribution.png"))
                plt.close()

                # Heatmap before
                plt.figure(figsize=(8, 6))
                sns.heatmap(param_b.cpu().numpy(), cmap='coolwarm', center=0)
                plt.title(f"Weight Heatmap Before ({ds} - {name_b})")
                plt.xlabel("Columns")
                plt.ylabel("Rows")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{ds}_{name_b.replace('.', '_')}_heatmap_before.png"))
                plt.close()

                # Heatmap after
                plt.figure(figsize=(8, 6))
                sns.heatmap(param_a.cpu().numpy(), cmap='coolwarm', center=0)
                plt.title(f"Weight Heatmap After ({ds} - {name_a})")
                plt.xlabel("Columns")
                plt.ylabel("Rows")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{ds}_{name_a.replace('.', '_')}_heatmap_after.png"))
                plt.close()

                # Heatmap difference
                plt.figure(figsize=(8, 6))
                diff = (param_a - param_b).cpu().numpy()
                sns.heatmap(diff, cmap='coolwarm', center=0)
                plt.title(f"Weight Heatmap Difference ({ds} - {name_b})")
                plt.xlabel("Columns")
                plt.ylabel("Rows")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{ds}_{name_b.replace('.', '_')}_heatmap_diff.png"))
                plt.close()


import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_distributions_svd(classification_head_before, classification_head_after, ds, save_dir):
    """
    Compare weight distributions and apply SVD analysis for classification_head_before and classification_head_after,
    then plot and save the figures to the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for (name_b, param_b), (name_a, param_a) in zip(
            classification_head_before.items(),
            classification_head_after.items(),
        ):
            if 'weight' in name_b and param_b.dim() == 2:
                # SVD analysis for before weights
                u_b, s_b, v_b = np.linalg.svd(param_b.cpu().numpy(), full_matrices=False)
                # SVD analysis for after weights
                u_a, s_a, v_a = np.linalg.svd(param_a.cpu().numpy(), full_matrices=False)
                
                # Plotting the singular values for before and after
                plt.figure(figsize=(10, 4))
                plt.plot(s_b, label='Singular values Before', color='blue', alpha=0.7)
                plt.plot(s_a, label='Singular values After', color='red', alpha=0.7)
                plt.xlabel("Index")
                plt.ylabel("Singular Value")
                plt.title(f"SVD Singular Values ({ds} - {name_b})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{ds}_{name_b.replace('.', '_')}_svd_singular_values.png"))
                plt.close()

                # SVD difference (before - after)
                s_diff = s_a - s_b
                plt.figure(figsize=(10, 4))
                plt.plot(s_diff, label='Singular Value Difference', color='purple', alpha=0.7)
                plt.xlabel("Index")
                plt.ylabel("Difference in Singular Value")
                plt.title(f"SVD Singular Value Difference ({ds} - {name_b})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{ds}_{name_b.replace('.', '_')}_svd_singular_value_diff.png"))
                plt.close()