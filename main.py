from test_routines import run_data_tests
from train import train_model
from utils import plot_heat_diffusion
from inspect_model import inspect_model

if __name__ == "__main__":
    print("=== Phase 1: Executing Test Suite ===")
    run_data_tests()
    
    print("=== Phase 2: Model Training ===")
    # train_model() # Uncomment to execute the full training loop
    
    print("=== Phase 3: Reporting & Visualisation ===")
    inspect_model()
    plot_heat_diffusion()