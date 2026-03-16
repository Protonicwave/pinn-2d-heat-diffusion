import torch
from data_generator import HeatDataGenerator
from rich.console import Console

console = Console()

def run_data_tests():
    console.print("[bold yellow]Executing Unit Tests: Data Generator[/bold yellow]")
    generator = HeatDataGenerator()
    
    # Test 1: Verify gradient tracking on interior points
    x_int, y_int, t_int = generator.generate_interior_points(100)
    assert x_int.requires_grad, "FAIL: Gradient tracking disabled for X."
    assert y_int.requires_grad, "FAIL: Gradient tracking disabled for Y."
    assert t_int.requires_grad, "FAIL: Gradient tracking disabled for T."
    
    # Test 2: Verify boundary condition enforcement
    x_b, y_b, t_b, u_b = generator.generate_boundary_points(100)
    assert torch.all(u_b == 0.0), "FAIL: Boundary targets do not equal 0."
    assert x_b.shape == (100, 1), f"FAIL: Shape mismatch. Expected (100, 1), got {x_b.shape}."
    
    # Test 3: Verify initial condition temporal alignment
    x_i, y_i, t_i, u_i = generator.generate_initial_points(100)
    assert torch.all(t_i == 0.0), "FAIL: Initial conditions not locked to t=0."
    
    console.print("[bold green]✓ All data generation tests passed successfully.[/bold green]\n")

if __name__ == "__main__":
    run_data_tests()