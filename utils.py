import torch
import numpy as np
import matplotlib.pyplot as plt
from network import PINN
from rich.console import Console

console = Console()

def plot_heat_diffusion(model_path: str = "pinn_heat_model.pth"):
    """Loads trained weights and generates a temporal heatmap plot for the report."""
    model = PINN()
    try:
        model.load_state_dict(torch.load(model_path))
        # Disable dropout/batchnorm and gradient tracking for evaluation
        model.eval() 
        console.print("[green]✓ Model weights loaded successfully.[/green]")
    except FileNotFoundError:
        console.print("[red]✗ Error: Model file missing. Execute train.py first.[/red]")
        return

    # Generate a 100x100 resolution grid for inference
    x_vals = np.linspace(0, 1, 100)
    y_vals = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    time_steps = [0.0, 0.1, 0.5]
    
    # Disable autograd to optimise memory during inference
    with torch.no_grad():
        for i, t_val in enumerate(time_steps):
            t_tensor = torch.full_like(x_tensor, t_val)
            u_pred = model(x_tensor, y_tensor, t_tensor)
            
            # Reshape 1D predictions back into the 2D grid
            U = u_pred.numpy().reshape(100, 100)
            
            ax = axes[i]
            c = ax.contourf(X, Y, U, levels=50, cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f"Time $t = {t_val}$")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')

    # Append a unified colourbar to the figure
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(c, cax=cbar_ax, label="Temperature (u)")

    plt.suptitle("PINN Prediction: 2D Heat Diffusion Over Time", fontsize=16)
    plt.savefig("heat_diffusion_results.png", dpi=300, bbox_inches='tight')
    console.print("[bold green]✓ Visualisation saved to 'heat_diffusion_results.png'[/bold green]")
    plt.show()

if __name__ == "__main__":
    plot_heat_diffusion()