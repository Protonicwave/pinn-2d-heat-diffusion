# Physics-Informed Neural Network (PINN): Solving the 2D Heat Equation

As a Physics and Computer Science undergrad, I've always felt that standard Machine Learning treats the physical world like a black box. You feed an AI a massive dataset, and it magically guesses the output. But what if we don't have a massive dataset? What if, instead, we have the unbending mathematical rules of the universe?

This project is my deep dive into Physics-Informed Neural Networks (PINNs). Instead of training an AI on thousands of pre-calculated simulations, I built a continuous neural network that learns to simulate the diffusion of heat across a 2D metal plate entirely from scratch, guided only by the laws of thermodynamics and PyTorch's automatic differentiation engine.

## The Core Concept: Math Over Data
Traditional neural networks map discrete data points. This network is designed to act as a continuous mathematical function mapping spatial coordinates $(x, y)$ and time $t$ to a temperature $u$.To do this, the network must obey the 2D Heat Equation (a Partial Differential Equation):

$$\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

Instead of a standard Mean Squared Error against a dataset, I engineered a custom loss function that heavily penalises the AI if its predictions break this mathematical law.

## Software Architecture & Engineering
I didn't want to just write a messy, linear Jupyter Notebook. I wanted to build a robust, modular, and testable piece of software.

* **Object-Oriented & Modular:** The physics engine, data generation, and network architecture are completely decoupled. You can swap out the PDE or the boundary conditions without rewriting the training loop.

* **Continuous Activation:** I explicitly avoided standard ReLU activations. Because the Heat Equation requires calculating second-order derivatives, the network uses smooth Tanh activations to ensure the calculus doesn't break at sharp corners.

* **Automated Testing:** The repository includes a test_routines.py suite using standard assertions to mathematically verify that the generated boundary and initial conditions strictly obey the rules of the simulation before any training begins.

## The Optimisation Strategy (Adam + L-BFGS)
Training a PINN involves navigating a highly complex, rugged loss landscape because the AI is balancing multiple conflicting goals (Boundary Data Loss vs. Physics Residual Loss).

Standard ML optimisers fail here. To solve this, I implemented a two-stage optimisation strategy:

1. **First-Order Descent (Adam):** The network trains for the first 1,000 epochs using Adam to rapidly descend the highest peaks of the loss landscape.
2. **Second-Order Precision (L-BFGS):** Once in the local valley, the training hands off to L-BFGS. While computationally heavier per step, this optimiser evaluates the curvature (second derivative) of the loss landscape, allowing it to mathematically pinpoint the exact global minimum and crush the physics loss to near absolute zero.

## Results & Visualisation
By the end of the training cycle, the AI successfully learned to model the heat diffusion over time without ever seeing a "correct" simulation dataset.

(Note: Run `utils.py` to generate the high-resolution publication-ready heatmaps from the trained `.pth` weights)

## How to Run the Project
1. **Clone the repository and install dependencies:**
```bash
git clone [https://github.com/Protonicwave/pinn-heat-equation.git](https://github.com/yourusername/pinn-heat-equation.git)
cd pinn-heat-equation
pip install torch numpy matplotlib rich
```
2. **Run the Unit Tests (Always test first!):**
```bash
python test_routines.py
```
3. **Train the Model:**
```bash
python train.py
```
4. **Generate the Visualisation & Inspect Parameters**

```bash
python main.py
```

## Project Structure
* `data_generator.py`: Generates the spatial/temporal collocation points and handles boundary/initial conditions.
* `network.py`: The continuous, fully connected neural network architecture.
* `physics.py`: The `autograd` engine that calculates partial derivatives and the custom PDE loss function.
* `train.py`: The orchestrator handling the two-stage optimisation loop.
* `utils.py`: Matplotlib plotting utilities for reproducibility.
* `inspect_model.py`: A benchmarking script to calculate and log the trainable parameter overhead.
* `test_routines.py`: The verification suite for the physics rules.
