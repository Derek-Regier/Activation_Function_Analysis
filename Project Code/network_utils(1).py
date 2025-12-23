"""
Neural Network Utilities

This module provides functions for:
- Data generation (binary classification and 2D classification)
- Single-layer network training
- Multi-layer network (MLP) training
- Prediction functions
"""

import numpy as np
from activation_functions import ActivationFunctions


class DataGenerator:
    """Generate synthetic datasets for neural network training"""
    
    @staticmethod
    def generate_binary_classification(w, bias, n_samples=1000, noise=0.0):
        """
        Generate binary classification dataset
        
        Args:
            w: weight vector [w1, w2, w3]
            bias: bias term
            n_samples: number of samples to generate
            noise: noise level for stochastic generation
            
        Returns:
            X: input features (n_samples x 3)
            y: binary labels (n_samples x 1)
        """
        X = np.random.rand(n_samples, 3)
        z = X @ w + bias + np.random.randn(n_samples) * noise
        
        # Apply sigmoid to convert to probability, then threshold
        prob = 1 / (1 + np.exp(-z))
        y = (prob > 0.5).astype(float).reshape(-1, 1)
        
        return X, y
    
    @staticmethod
    def generate_2d_classification(n_samples=1000, seed=None):
        """
        Generate 2D non-linearly separable dataset (concentric circles)
        
        Args:
            n_samples: number of samples
            seed: random seed for reproducibility
            
        Returns:
            X: input features (n_samples x 2)
            y: binary labels (n_samples x 1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_per_class = n_samples // 2
        
        # Class 0: inner circle
        theta_0 = 2 * np.pi * np.random.rand(n_per_class)
        r_0 = 0.5 * np.random.rand(n_per_class)
        X_0 = np.column_stack([r_0 * np.cos(theta_0), r_0 * np.sin(theta_0)])
        y_0 = np.zeros((n_per_class, 1))
        
        # Class 1: outer circle
        theta_1 = 2 * np.pi * np.random.rand(n_per_class)
        r_1 = 0.5 + 0.5 * np.random.rand(n_per_class)
        X_1 = np.column_stack([r_1 * np.cos(theta_1), r_1 * np.sin(theta_1)])
        y_1 = np.ones((n_per_class, 1))
        
        # Combine and shuffle
        X = np.vstack([X_0, X_1])
        y = np.vstack([y_0, y_1])
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        return X, y


class SingleLayerNetwork:
    """Single-layer neural network"""
    
    def __init__(self, input_size=3, activation='sigmoid', learning_rate=0.1):
        """
        Initialize single-layer network
        
        Args:
            input_size: number of input features
            activation: activation function name
            learning_rate: learning rate for gradient descent
        """
        self.input_size = input_size
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Initialize weights and bias
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
    
    def forward(self, X):
        """
        Forward pass
        
        Args:
            X: input features (n_samples x input_size)
            
        Returns:
            z: pre-activation (n_samples,)
            a: activated output (n_samples,)
        """
        z = X @ self.weights + self.bias
        a = ActivationFunctions.activate(z, self.activation)
        return z, a
    
    def backward(self, X, y, z, a):
        """
        Backward pass - compute gradients
        
        Args:
            X: input features
            y: target labels
            z: pre-activation values
            a: activated values
            
        Returns:
            dw: weight gradients
            db: bias gradient
        """
        m = X.shape[0]
        
        # Activation derivative
        dact = ActivationFunctions.activate_derivative(z, self.activation)
        
        # Error
        error = a - y.flatten()
        
        # Gradients
        dw = (X.T @ (error * dact)) / m
        db = np.mean(error * dact)
        
        return dw, db
    
    def train(self, X, y, epochs=100, verbose=True):
        """
        Train network with gradient descent
        
        Args:
            X: training features
            y: training labels
            epochs: number of training epochs
            verbose: whether to print progress
            
        Returns:
            losses: list of training losses
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            z, a = self.forward(X)
            
            # Clamp activations to valid probability range for loss computation
            a_safe = np.clip(a, 1e-10, 1 - 1e-10)
            
            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(a_safe) + (1 - y) * np.log(1 - a_safe))
            losses.append(loss)
            
            # Backward pass
            dw, db = self.backward(X, y, z, a)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: input features
            
        Returns:
            predictions: binary predictions (0 or 1)
            probabilities: probability scores
        """
        _, a = self.forward(X)
        predictions = (a > 0.5).astype(int)
        return predictions, a
    
    def accuracy(self, X, y):
        """
        Compute accuracy
        
        Args:
            X: input features
            y: target labels
            
        Returns:
            accuracy: fraction of correct predictions
        """
        predictions, _ = self.predict(X)
        return np.mean(predictions == y.flatten())


class MultiLayerNetwork:
    """Multi-layer perceptron (MLP) with one hidden layer"""
    
    def __init__(self, input_size=2, hidden_size=5, output_size=1,
                 hidden_activation='sigmoid', output_activation='sigmoid',
                 learning_rate=0.1):
        """
        Initialize 2-layer network
        
        Args:
            input_size: number of input features
            hidden_size: number of hidden units
            output_size: number of output units
            hidden_activation: activation for hidden layer
            output_activation: activation for output layer
            learning_rate: learning rate
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        
        # Initialize weights (He initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """
        Forward pass
        
        Args:
            X: input features (n_samples x input_size)
            
        Returns:
            z1, h, z2, output: intermediate values for backprop
        """
        # Hidden layer
        z1 = X @ self.W1 + self.b1
        h = ActivationFunctions.activate(z1, self.hidden_activation)
        
        # Output layer
        z2 = h @ self.W2 + self.b2
        output = ActivationFunctions.activate(z2, self.output_activation)
        
        return z1, h, z2, output
    
    def backward(self, X, y, z1, h, z2, output):
        """
        Backward pass with backpropagation
        
        Args:
            X: input features
            y: target labels
            z1, h, z2, output: values from forward pass
            
        Returns:
            dW1, db1, dW2, db2: gradients
        """
        m = X.shape[0]
        
        # Output layer error
        dz2 = output - y
        dz2 = dz2 * ActivationFunctions.activate_derivative(z2, self.output_activation)
        
        # Gradients for output layer
        dW2 = (h.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer error (backprop)
        dh = dz2 @ self.W2.T
        dz1 = dh * ActivationFunctions.activate_derivative(z1, self.hidden_activation)
        
        # Gradients for hidden layer
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def train(self, X, y, epochs=100, verbose=True):
        """
        Train network
        
        Args:
            X: training features
            y: training labels
            epochs: number of epochs
            verbose: whether to print progress
            
        Returns:
            losses: training losses
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            z1, h, z2, output = self.forward(X)
            
            # Clamp output to valid probability range for loss computation
            output_safe = np.clip(output, 1e-10, 1 - 1e-10)
            
            # Compute loss
            loss = -np.mean(y * np.log(output_safe) + 
                          (1 - y) * np.log(1 - output_safe))
            losses.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y, z1, h, z2, output)
            
            # Update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: input features
            
        Returns:
            predictions: binary predictions
            probabilities: probability scores
        """
        _, _, _, output = self.forward(X)
        predictions = (output > 0.5).astype(int)
        return predictions.flatten(), output.flatten()
    
    def accuracy(self, X, y):
        """
        Compute accuracy
        
        Args:
            X: input features
            y: target labels
            
        Returns:
            accuracy: fraction of correct predictions
        """
        predictions, _ = self.predict(X)
        return np.mean(predictions == y.flatten())