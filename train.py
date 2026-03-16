import torch
import torch.optim as optim
from network import PINN
from physics import PINNPhysicsEngine
from data_generator import HeatDataGenerator
from rich.console import Console
from rich.panel import Panel

console = Console()

def train_model():
    console.print(Panel.fit("[bold yellow]Initialising PINN Training Pipeline[/bold yellow]"))

    generator = HeatDataGenerator()
    model = PINN()
    engine = PINNPhysicsEngine(alpha=0.01)

    x_int, y_int, t_int = generator.generate_interior_points(2000)
    x_b, y_b, t_b, u_b = generator.generate_boundary_points(500)
    x_i, y_i, t_i, u_i = generator.generate_initial_points(500)

    # --- Stage 1: Adam Optimisation ---
    # Used for rapid initial descent across the loss landscape.
    console.print("[bold cyan]Stage 1: Adam Optimisation[/bold cyan]")
    adam_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs_adam = 1000

    for epoch in range(epochs_adam):
        adam_optimizer.zero_grad()
        
        loss_physics = engine.compute_physics_loss(model, x_int, y_int, t_int)
        loss_boundary = engine.compute_data_loss(model, x_b, y_b, t_b, u_b)
        loss_initial = engine.compute_data_loss(model, x_i, y_i, t_i, u_i)
        
        loss_total = loss_physics + loss_boundary + loss_initial
        loss_total.backward()
        adam_optimizer.step()
        
        if epoch % 200 == 0:
            console.print(f"Adam Epoch {epoch:04d} | Total Loss: {loss_total.item():.6f}")

    # --- Stage 2: L-BFGS Optimisation ---
    # A second-order optimiser used for precise convergence to the local minimum.
    console.print("[bold cyan]Stage 2: L-BFGS Optimisation[/bold cyan]")
    lbfgs_optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500)

    def closure():
        """Closure function required by L-BFGS to re-evaluate the loss multiple times per step."""
        lbfgs_optimizer.zero_grad()
        loss_total = (
            engine.compute_physics_loss(model, x_int, y_int, t_int) +
            engine.compute_data_loss(model, x_b, y_b, t_b, u_b) +
            engine.compute_data_loss(model, x_i, y_i, t_i, u_i)
        )
        loss_total.backward()
        return loss_total

    lbfgs_optimizer.step(closure)
    
    console.print(f"\n[bold green]✓ Training Complete! Final Total Loss: {closure().item():.8f}[/bold green]")
    torch.save(model.state_dict(), "pinn_heat_model.pth")

if __name__ == "__main__":
    train_model()