"""
Main Analysis Script

Comprehensive analysis of activation functions including:
1. Single-layer network comparison
2. Activation function visualization
3. Multi-layer network training
4. Vanishing gradient visualization
5. Decision boundary visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from activation_functions import ActivationFunctions
from network_utils import DataGenerator, SingleLayerNetwork, MultiLayerNetwork
from visualizations import ActivationVisualizer


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    print("=" * 80)
    print("ACTIVATION FUNCTIONS IN NEURAL NETWORKS - ANALYSIS")
    print("=" * 80)
    print()

    # ============================================================================
    # PART 1: SINGLE-LAYER NETWORK COMPARISON
    # ============================================================================
    print("PART 1: Single-Layer Network Training with Different Activations")
    print("-" * 80)

    # Generate data
    w_true = np.array([0.25, 0.35, 0.40])
    bias_true = 0.65
    X, y = DataGenerator.generate_binary_classification(
        w_true, bias_true, n_samples=5000
    )

    # Train networks with different activations
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu']
    results = []
    training_histories = []

    print(f"\n{'Activation':<15} | {'Final Loss':<12} | {'Accuracy':<12}")
    print("-" * 45)

    for activation in activations:
        net = SingleLayerNetwork(
            input_size=3,
            activation=activation,
            learning_rate=0.1
        )
        losses = net.train(X, y, epochs=50, verbose=False)
        training_histories.append(losses)

        accuracy = net.accuracy(X, y)
        final_loss = losses[-1]

        results.append({
            'activation': activation,
            'final_loss': final_loss,
            'accuracy': accuracy,
            'losses': losses
        })

        print(f"{activation:<15} | {final_loss:<12.6f} | {accuracy:<12.4f}")

    print()

    # ============================================================================
    # PART 2: ACTIVATION FUNCTION VISUALIZATION
    # ============================================================================
    print("\nPART 2: Generating Activation Function Visualizations")

    ActivationVisualizer.plot_activation_functions(
        save_path='activation_functions.png'
    )
    print("Saved: activation_functions.png")

    ActivationVisualizer.plot_vanishing_gradient(
        save_path='vanishing_gradient.png'
    )
    print("Saved: vanishing_gradient.png")

    ActivationVisualizer.plot_training_curves(
        training_histories,
        activations,
        save_path='training_curves_single.png'
    )
    print("Saved: training_curves_single.png")

    print()

    # ============================================================================
    # PART 3: MULTI-LAYER NETWORK COMPARISON
    # ============================================================================
    print("\nPART 3: Multi-Layer Network (MLP) Training")

    X_2d, y_2d = DataGenerator.generate_2d_classification(
        n_samples=2000,
        seed=42
    )

    hidden_activations = ['sigmoid', 'tanh', 'relu']
    mlp_histories = []
    mlp_networks = []

    print(f"\n{'Hidden Act.':<15} | {'Final Loss':<12} | {'Accuracy':<12}")

    for hidden_act in hidden_activations:
        mlp = MultiLayerNetwork(
            input_size=2,
            hidden_size=10,
            output_size=1,
            hidden_activation=hidden_act,
            output_activation='sigmoid',
            learning_rate=0.1
        )

        losses = mlp.train(X_2d, y_2d, epochs=1000, verbose=False)
        mlp_histories.append(losses)
        mlp_networks.append(mlp)

        accuracy = mlp.accuracy(X_2d, y_2d)
        final_loss = losses[-1]

        print(f"{hidden_act:<15} | {final_loss:<12.6f} | {accuracy:<12.4f}")

    print()

    ActivationVisualizer.plot_mlp_training_curves(
        mlp_histories,
        hidden_activations,
        save_path='training_curves_mlp.png'
    )
    print("Saved: training_curves_mlp.png")

    # ============================================================================
    # PART 4: DECISION BOUNDARIES
    # ============================================================================
    print("\nPART 4: Decision Boundary Visualization\n")

    for hidden_act, mlp in zip(hidden_activations, mlp_networks):
        filename = f'decision_boundary_{hidden_act}.png'
        title = f'MLP Decision Boundary ({hidden_act.capitalize()} Hidden → Sigmoid Output)'
        ActivationVisualizer.plot_decision_boundary(
            mlp,
            X_2d,
            y_2d,
            title=title,
            save_path=filename
        )
        print(f"Saved: {filename}")

    print()

    # ============================================================================
    # PART 5: GRADIENT ANALYSIS
    # ============================================================================
    print("\nPART 5: Gradient Magnitude Through Layers\n")

    z = np.random.randn(1000, 10)
    depths = [5, 10, 20, 50]

    print("Gradient magnitude after backpropagating through n layers:\n")
    print(f"{'Depth':<10} | {'Sigmoid':<15} | {'Tanh':<15} | {'ReLU':<15}")

    for depth in depths:
        grad_sig = 1.0
        grad_tanh = 1.0
        grad_relu = 1.0

        for _ in range(depth):
            grad_sig *= ActivationFunctions.sigmoid_derivative(z).mean()
            grad_tanh *= ActivationFunctions.tanh_derivative(z).mean()
            grad_relu *= ActivationFunctions.relu_derivative(z).mean()

        print(
            f"{depth:<10} | "
            f"{grad_sig:<15.2e} | "
            f"{grad_tanh:<15.2e} | "
            f"{grad_relu:<15.2e}"
        )

    print()

    # Show plots (interactive environments)
    plt.show()


if __name__ == "__main__":
    main()
