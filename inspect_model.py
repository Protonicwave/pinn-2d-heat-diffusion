import torch
from network import PINN
from rich.console import Console
from rich.table import Table

console = Console()

def inspect_model(model_path="pinn_heat_model.pth"):
    """Parses the trained model state dictionary to quantify trainable parameters."""
    model = PINN()
    
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        console.print(f"[red]✗ Error: Could not locate '{model_path}'.[/red]")
        return

    table = Table(title="Neural Network Parameter Architecture", show_header=True)
    table.add_column("Layer Identification", style="cyan")
    table.add_column("Tensor Shape", style="green")
    table.add_column("Parameter Count", justify="right", style="yellow")

    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        
        param_count = parameter.numel() 
        total_params += param_count
        shape_str = str(list(parameter.shape))
        table.add_row(name, shape_str, f"{param_count:,}")

    console.print(table)
    console.print(f"[bold green]Aggregate Trainable Parameters: {total_params:,}[/bold green]")

if __name__ == "__main__":
    inspect_model()