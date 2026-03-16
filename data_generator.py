import torch
from rich.console import Console
from rich.panel import Panel

console = Console()

class HeatDataGenerator:
    """
    Generates spatial (x, y) and temporal (t) collocation points for training 
    a 2D Heat Equation Physics-Informed Neural Network (PINN).
    """
    def __init__(self, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, t_min=0.0, t_max=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_min = t_min
        self.t_max = t_max
        
        console.print(Panel.fit(
            "[bold blue]HeatDataGenerator Initialised[/bold blue]\n"
            f"Spatial Domain: X[{self.x_min}, {self.x_max}], Y[{self.y_min}, {self.y_max}]\n"
            f"Time Domain: T[{self.t_min}, {self.t_max}]"
        ))

    def generate_interior_points(self, num_points: int):
        """Generates random collocation points inside the spatial domain to enforce the PDE."""
        x = torch.empty(num_points, 1).uniform_(self.x_min, self.x_max).requires_grad_(True)
        y = torch.empty(num_points, 1).uniform_(self.y_min, self.y_max).requires_grad_(True)
        t = torch.empty(num_points, 1).uniform_(self.t_min, self.t_max).requires_grad_(True)
        
        console.print(f"[green]✓ Generated {num_points} interior points[/green]")
        return x, y, t

    def generate_boundary_points(self, num_points: int):
        """Generates uniformly distributed points along the domain edges (u = 0)."""
        points_per_edge = num_points // 4
        
        # Define the four edges of the 2D plane
        x_bottom = torch.empty(points_per_edge, 1).uniform_(self.x_min, self.x_max)
        y_bottom = torch.full((points_per_edge, 1), self.y_min)
        
        x_top = torch.empty(points_per_edge, 1).uniform_(self.x_min, self.x_max)
        y_top = torch.full((points_per_edge, 1), self.y_max)
        
        x_left = torch.full((points_per_edge, 1), self.x_min)
        y_left = torch.empty(points_per_edge, 1).uniform_(self.y_min, self.y_max)
        
        x_right = torch.full((points_per_edge, 1), self.x_max)
        y_right = torch.empty(points_per_edge, 1).uniform_(self.y_min, self.y_max)
        
        x = torch.cat([x_bottom, x_top, x_left, x_right]).requires_grad_(True)
        y = torch.cat([y_bottom, y_top, y_left, y_right]).requires_grad_(True)
        t = torch.empty(num_points, 1).uniform_(self.t_min, self.t_max).requires_grad_(True)
        
        # Enforce boundary condition: Temperature is 0 at the edges
        u_target = torch.zeros_like(x)
        
        console.print(f"[green]✓ Generated {num_points} boundary points[/green]")
        return x, y, t, u_target

    def generate_initial_points(self, num_points: int):
        """Generates points at t=0 with a smooth Gaussian heat distribution."""
        x = torch.empty(num_points, 1).uniform_(self.x_min, self.x_max).requires_grad_(True)
        y = torch.empty(num_points, 1).uniform_(self.y_min, self.y_max).requires_grad_(True)
        t = torch.zeros(num_points, 1).requires_grad_(True)
        
        # Apply a Gaussian distribution to create a differentiable initial heat source
        u_target = torch.exp(-100 * ((x - 0.5)**2 + (y - 0.5)**2))
        
        console.print(f"[green]✓ Generated {num_points} initial points[/green]")
        return x, y, t, u_target