"""
Activation Functions and Their Derivatives

This module implements common activation functions used in neural networks:
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- ELU

Each activation function includes both the forward pass and derivative.
"""

import numpy as np


class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    # ========== SIGMOID ==========
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        Formula: σ(x) = 1 / (1 + e^(-x))
        Range: (0, 1)
        
        Args:
            x: input value or array
            
        Returns:
            Activated value(s)
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivative of sigmoid activation
        Formula: σ'(x) = σ(x)(1 - σ(x))
        
        Args:
            x: pre-activation value(s)
            
        Returns:
            Derivative value(s)
        """
        sig = ActivationFunctions.sigmoid(x)
        return sig * (1 - sig)
    
    # ========== TANH ==========
    @staticmethod
    def tanh_activation(x):
        """
        Hyperbolic tangent activation function
        Formula: tanh(x)
        Range: (-1, 1)
        
        Args:
            x: input value or array
            
        Returns:
            Activated value(s)
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """
        Derivative of tanh activation
        Formula: tanh'(x) = 1 - tanh^2(x)
        
        Args:
            x: pre-activation value(s)
            
        Returns:
            Derivative value(s)
        """
        return 1 - np.tanh(x) ** 2
    
    # ========== RELU ==========
    @staticmethod
    def relu(x):
        """
        Rectified Linear Unit activation
        Formula: ReLU(x) = max(0, x)
        Range: [0, ∞)
        
        Args:
            x: input value or array
            
        Returns:
            Activated value(s)
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """
        Derivative of ReLU
        Formula: ReLU'(x) = 1 if x > 0, else 0
        
        Args:
            x: pre-activation value(s)
            
        Returns:
            Derivative value(s)
        """
        return (x > 0).astype(float)
    
    # ========== LEAKY RELU ==========
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        Leaky Rectified Linear Unit activation
        Formula: LeakyReLU(x) = x if x > 0, else alpha*x
        Range: ℝ (all reals)
        
        Args:
            x: input value or array
            alpha: leak parameter (default 0.01)
            
        Returns:
            Activated value(s)
        """
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """
        Derivative of Leaky ReLU
        Formula: LeakyReLU'(x) = 1 if x > 0, else alpha
        
        Args:
            x: pre-activation value(s)
            alpha: leak parameter (default 0.01)
            
        Returns:
            Derivative value(s)
        """
        return np.where(x > 0, 1.0, alpha)
    
    # ========== ELU ==========
    @staticmethod
    def elu(x, alpha=1.0):
        """
        Exponential Linear Unit activation
        Formula: ELU(x) = x if x > 0, else alpha*(e^x - 1)
        Range: ≈ ℝ
        
        Args:
            x: input value or array
            alpha: scaling parameter (default 1.0)
            
        Returns:
            Activated value(s)
        """
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x, alpha=1.0):
        """
        Derivative of ELU
        Formula: ELU'(x) = 1 if x > 0, else alpha*e^x
        
        Args:
            x: pre-activation value(s)
            alpha: scaling parameter (default 1.0)
            
        Returns:
            Derivative value(s)
        """
        return np.where(x > 0, 1.0, alpha * np.exp(x))
    
    # ========== GENERIC WRAPPER ==========
    @staticmethod
    def activate(x, activation_type='sigmoid', alpha=0.01):
        """
        Generic activation function wrapper
        
        Args:
            x: input value or array
            activation_type: string ('sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu')
            alpha: parameter for leaky_relu and elu
            
        Returns:
            Activated value(s)
        """
        if activation_type == 'sigmoid':
            return ActivationFunctions.sigmoid(x)
        elif activation_type == 'tanh':
            return ActivationFunctions.tanh_activation(x)
        elif activation_type == 'relu':
            return ActivationFunctions.relu(x)
        elif activation_type == 'leaky_relu':
            return ActivationFunctions.leaky_relu(x, alpha)
        elif activation_type == 'elu':
            return ActivationFunctions.elu(x, alpha)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    @staticmethod
    def activate_derivative(x, activation_type='sigmoid', alpha=0.01):
        """
        Generic derivative wrapper
        
        Args:
            x: pre-activation value or array
            activation_type: string ('sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu')
            alpha: parameter for leaky_relu and elu
            
        Returns:
            Derivative value(s)
        """
        if activation_type == 'sigmoid':
            return ActivationFunctions.sigmoid_derivative(x)
        elif activation_type == 'tanh':
            return ActivationFunctions.tanh_derivative(x)
        elif activation_type == 'relu':
            return ActivationFunctions.relu_derivative(x)
        elif activation_type == 'leaky_relu':
            return ActivationFunctions.leaky_relu_derivative(x, alpha)
        elif activation_type == 'elu':
            return ActivationFunctions.elu_derivative(x, alpha)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")


# Convenience functions for direct access
def sigmoid(x):
    return ActivationFunctions.sigmoid(x)

def sigmoid_derivative(x):
    return ActivationFunctions.sigmoid_derivative(x)

def tanh_activation(x):
    return ActivationFunctions.tanh_activation(x)

def tanh_derivative(x):
    return ActivationFunctions.tanh_derivative(x)

def relu(x):
    return ActivationFunctions.relu(x)

def relu_derivative(x):
    return ActivationFunctions.relu_derivative(x)

def leaky_relu(x, alpha=0.01):
    return ActivationFunctions.leaky_relu(x, alpha)

def leaky_relu_derivative(x, alpha=0.01):
    return ActivationFunctions.leaky_relu_derivative(x, alpha)

def elu(x, alpha=1.0):
    return ActivationFunctions.elu(x, alpha)

def elu_derivative(x, alpha=1.0):
    return ActivationFunctions.elu_derivative(x, alpha)