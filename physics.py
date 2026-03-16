import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel

console = Console()

class PINNPhysicsEngine:
    """
    Calculates physical derivatives and enforces the 2D Heat Equation 
    using PyTorch's automatic differentiation.
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        
        console.print(Panel.fit(
            f"[bold cyan]Physics Engine Initialised[/bold cyan]\n"
            f"PDE: du/dt = alpha * (d2u/dx2 + d2u/dy2) | alpha={self.alpha}"
        ))

    def compute_gradients(self, u: torch.Tensor, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """Computes the first and second-order partial derivatives via autograd."""
        # create_graph=True allows for the computation of higher-order derivatives
        du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]
        
        return du_dt, d2u_dx2, d2u_dy2

    def compute_physics_loss(self, network, x_int: torch.Tensor, y_int: torch.Tensor, t_int: torch.Tensor):
        """Calculates the mean squared error of the PDE residual at interior points."""
        u_pred = network(x_int, y_int, t_int)
        du_dt, d2u_dx2, d2u_dy2 = self.compute_gradients(u_pred, x_int, y_int, t_int)
        
        # The residual represents the deviation from the exact laws of physics
        residual = du_dt - self.alpha * (d2u_dx2 + d2u_dy2)
        target_zero = torch.zeros_like(residual)
        
        return self.mse_loss(residual, target_zero)

    def compute_data_loss(self, network, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, u_target: torch.Tensor):
        """Calculates the MSE between the network predictions and defined boundary/initial conditions."""
        u_pred = network(x, y, t)
        return self.mse_loss(u_pred, u_target)