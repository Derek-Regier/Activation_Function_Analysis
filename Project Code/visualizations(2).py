"""
Visualization Module

This module provides visualization functions for:
- Activation functions and their derivatives
- Vanishing gradient problem
- Training curves
- 2D decision boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from activation_functions import ActivationFunctions


class ActivationVisualizer:
    """Visualize activation functions and their properties"""
    
    @staticmethod
    def plot_activation_functions(save_path=None):
        """
        Plot all activation functions and their derivatives
        
        Args:
            save_path: optional path to save figure
        """
        x = np.linspace(-5, 5, 1000)
        
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        fig.suptitle('Activation Functions and Their Derivatives', fontsize=16, fontweight='bold')
        
        activations = [
            ('sigmoid', 'Sigmoid: $\\sigma(x) = \\frac{1}{1+e^{-x}}$', 'blue'),
            ('tanh', 'Tanh: $\\tanh(x)$', 'red'),
            ('relu', 'ReLU: $max(0, x)$', 'green'),
            ('leaky_relu', 'Leaky ReLU: $\\alpha=0.01$', 'orange'),
            ('elu', 'ELU: $\\alpha=1.0$', 'purple'),
        ]
        
        for idx, (act_name, title, color) in enumerate(activations):
            # Forward pass
            y = ActivationFunctions.activate(x, act_name)
            axes[0, idx].plot(x, y, color=color, linewidth=2.5)
            axes[0, idx].grid(True, alpha=0.3)
            axes[0, idx].set_title(title, fontsize=11)
            axes[0, idx].set_ylabel(f'{act_name}(x)')
            axes[0, idx].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            axes[0, idx].axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            
            # Derivative
            dy = ActivationFunctions.activate_derivative(x, act_name)
            axes[1, idx].plot(x, dy, color=color, linewidth=2.5)
            axes[1, idx].grid(True, alpha=0.3)
            axes[1, idx].set_title(f"Derivative: ${act_name}'(x)$", fontsize=11)
            axes[1, idx].set_ylabel(f"{act_name}'(x)")
            axes[1, idx].set_xlabel('x')
            axes[1, idx].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            axes[1, idx].axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_vanishing_gradient(save_path=None):
        """
        Visualize the vanishing gradient problem
        
        Args:
            save_path: optional path to save figure
        """
        x = np.linspace(-5, 5, 1000)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle('Vanishing Gradient Problem', fontsize=14, fontweight='bold')
        
        # Sigmoid
        grad_sigmoid = ActivationFunctions.sigmoid_derivative(x)
        axes[0].plot(x, grad_sigmoid, 'b-', linewidth=2.5)
        axes[0].fill_between(x, grad_sigmoid, alpha=0.3, color='blue')
        axes[0].set_title('Sigmoid: Severe Vanishing', fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Gradient $\\sigma'(x)$", fontsize=11)
        axes[0].set_xlabel('x')
        axes[0].grid(True, alpha=0.3)
        axes[0].text(0, 0.22, f'Max: {grad_sigmoid.max():.3f}', fontsize=10, 
                    ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Tanh
        grad_tanh = ActivationFunctions.tanh_derivative(x)
        axes[1].plot(x, grad_tanh, 'r-', linewidth=2.5)
        axes[1].fill_between(x, grad_tanh, alpha=0.3, color='red')
        axes[1].set_title('Tanh: Moderate Vanishing', fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Gradient $\\tanh'(x)$", fontsize=11)
        axes[1].set_xlabel('x')
        axes[1].grid(True, alpha=0.3)
        axes[1].text(0, 0.85, f'Max: {grad_tanh.max():.3f}', fontsize=10,
                    ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ReLU and Leaky ReLU
        grad_relu = ActivationFunctions.relu_derivative(x)
        grad_lrelu = ActivationFunctions.leaky_relu_derivative(x, 0.01)
        
        axes[2].plot(x, grad_relu, 'k-', linewidth=2.5, label='ReLU', alpha=0.8)
        axes[2].plot(x, grad_lrelu, 'g--', linewidth=2.5, label='Leaky ReLU')
        axes[2].fill_between(x, grad_relu, alpha=0.2, color='black')
        axes[2].fill_between(x, grad_lrelu, alpha=0.2, color='green')
        axes[2].set_title('ReLU: No Vanishing ✓', fontsize=12, fontweight='bold', color='darkgreen')
        axes[2].set_ylabel("Gradient", fontsize=11)
        axes[2].set_xlabel('x')
        axes[2].legend(fontsize=10, loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig

    
    @staticmethod
    def plot_training_curves(histories, activation_names, save_path=None):
        """
        Plot training curves for different activations
        
        Args:
            histories: list of loss histories
            activation_names: list of activation function names
            save_path: optional path to save figure
        """
        fig, axes = plt.subplots(1, len(histories), figsize=(15, 4))
        
        if len(histories) == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, (losses, name, color) in enumerate(zip(histories, activation_names, colors)):
            axes[idx].plot(losses, color=color, linewidth=2)
            axes[idx].fill_between(range(len(losses)), losses, alpha=0.3, color=color)
            axes[idx].set_title(f'{name.capitalize()} Activation', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Binary Cross-Entropy Loss')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([min(losses) * 0.9, max(losses) * 1.1])
        
        fig.suptitle('Single-Layer Network Training Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_mlp_training_curves(histories, activation_names, save_path=None):
        """
        Plot MLP training curves
        
        Args:
            histories: list of loss histories
            activation_names: list of hidden activation names
            save_path: optional path to save figure
        """
        fig, axes = plt.subplots(1, len(histories), figsize=(15, 4))
        
        if len(histories) == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green']
        
        for idx, (losses, name, color) in enumerate(zip(histories, activation_names, colors)):
            axes[idx].plot(losses, color=color, linewidth=2.5)
            axes[idx].fill_between(range(len(losses)), losses, alpha=0.3, color=color)
            axes[idx].set_title(f'2-Layer MLP\n{name.capitalize()} Hidden', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Binary Cross-Entropy Loss')
            axes[idx].grid(True, alpha=0.3)
        
        fig.suptitle('Multi-Layer Network Training (ReLU Hidden → Sigmoid Output)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_decision_boundary(network, X, y, title='Decision Boundary', save_path=None):
        """
        Plot 2D decision boundary for MLP
        
        Args:
            network: trained network
            X: input features (for range)
            y: labels (for coloring)
            title: plot title
            save_path: optional path to save figure
        """
        # Create mesh
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = np.c_[xx.ravel(), yy.ravel()]
        predictions, _ = network.predict(Z)
        Z = predictions.reshape(xx.shape)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#CCCCFF', '#FFCCCC'], alpha=0.6)
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='RdBu', 
                           edgecolors='black', linewidth=1, s=100, alpha=0.8)
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        blue_patch = mpatches.Patch(color='#CCCCFF', label='Predicted Class 0')
        red_patch = mpatches.Patch(color='#FFCCCC', label='Predicted Class 1')
        ax.legend(handles=[blue_patch, red_patch], loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig