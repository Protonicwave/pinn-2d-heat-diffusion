import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel

console = Console()

class PINN(nn.Module):
    """
    Fully connected neural network mapping spatial and temporal coordinates (x, y, t) 
    to predicted temperature (u).
    """
    def __init__(self, num_hidden_layers: int = 4, neurons_per_layer: int = 50):
        super(PINN, self).__init__()
        
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        
        # Input maps (x, y, t) to the first hidden layer
        self.input_layer = nn.Linear(3, self.neurons_per_layer)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.neurons_per_layer, self.neurons_per_layer) 
            for _ in range(self.num_hidden_layers)
        ])
        
        # Output maps the final hidden layer to a scalar temperature value (u)
        self.output_layer = nn.Linear(self.neurons_per_layer, 1)
        
        # Tanh is used over ReLU to ensure the continuous, non-zero second 
        # derivatives required by the Heat Equation PDE.
        self.activation = nn.Tanh()
        
        console.print(Panel.fit(
            f"[bold magenta]PINN Architecture Initialised[/bold magenta]\n"
            f"Inputs: 3 (x, y, t) -> Hidden: {self.num_hidden_layers}x{self.neurons_per_layer} -> Output: 1 (u)\n"
            f"Activation: Tanh"
        ))

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Executes the forward pass of the network."""
        # Concatenate inputs along the feature dimension
        inputs = torch.cat([x, y, t], dim=1)
        
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
            
        # Linear activation on the final output layer
        return self.output_layer(out)