"""
Neural Network Implementation

A simple neural network for multi-class classification built from scratch using only NumPy.
"""

import numpy as np
import random
import math


class MultiClassNeuralNetwork:
    """
    A simple neural network for multi-class classification.
    
    Architecture:
    - Input layer: 64 neurons (8x8 image pixels)
    - Hidden layer: 32 neurons with ReLU activation
    - Output layer: 10 neurons with Softmax activation
    """
    
    def __init__(self, n_input=64, n_hidden=32, n_output=10, learning_rate=0.1, seed=42):
        """
        Initialize the neural network.
        
        Args:
            n_input (int): Number of input features
            n_hidden (int): Number of hidden neurons
            n_output (int): Number of output classes
            learning_rate (float): Learning rate for gradient descent
            seed (int): Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights and biases using He and Xavier initialization."""
        # He initialization for ReLU
        self.W1 = np.random.randn(self.n_hidden, self.n_input) * math.sqrt(2/self.n_input)
        self.b1 = np.full((self.n_hidden, 1), 0.01)
        
        # Xavier initialization for output layer
        self.W2 = np.random.randn(self.n_output, self.n_hidden) * math.sqrt(1/self.n_hidden)
        self.b2 = np.full((self.n_output, 1), 0.01)
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function."""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function for multi-class output."""
        e = np.exp(x - np.max(x))  # Numerical stability
        return e / np.sum(e)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (np.array): Input data of shape (n_features, 1)
            
        Returns:
            tuple: (output probabilities, cache for backpropagation)
        """
        # Ensure input is column vector
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        # Forward pass
        h1 = self.W1 @ x + self.b1
        z1 = self.relu(h1)
        h2 = self.W2 @ z1 + self.b2
        y_hat = self.softmax(h2)
        
        # Cache for backpropagation
        cache = {
            "x": x,
            "h1": h1,
            "z1": z1,
            "h2": h2,
            "y_hat": y_hat
        }
        
        return y_hat, cache
    
    def cross_entropy_loss(self, y_hat, y_true):
        """
        Compute cross-entropy loss for multi-class classification.
        
        Args:
            y_hat (np.array): Predicted probabilities
            y_true (np.array): True one-hot encoded labels
            
        Returns:
            float: Cross-entropy loss
        """
        eps = 1e-9  # Small value to avoid log(0)
        return -float(np.sum(y_true * np.log(y_hat + eps)))
    
    def backward(self, y_true, cache):
        """
        Backward pass (backpropagation).
        
        Args:
            y_true (np.array): True one-hot encoded labels
            cache (dict): Cache from forward pass
            
        Returns:
            tuple: Gradients for all parameters
        """
        x = cache["x"]
        h1 = cache["h1"]
        z1 = cache["z1"]
        h2 = cache["h2"]
        y_hat = cache["y_hat"]
        
        # Output layer gradients
        dL_dh2 = y_hat - y_true
        dL_dW2 = np.dot(dL_dh2, z1.T)
        dL_db2 = dL_dh2
        
        # Hidden layer gradients
        dL_dz1 = np.dot(self.W2.T, dL_dh2)
        dL_dh1 = dL_dz1 * self.relu_derivative(h1)
        dL_dW1 = np.dot(dL_dh1, x.T)
        dL_db1 = dL_dh1
        
        return dL_dW1, dL_db1, dL_dW2, dL_db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        """Update network parameters using gradient descent."""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X_train, Y_train, epochs=100, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train (np.array): Training features
            Y_train (np.array): Training labels (one-hot encoded)
            epochs (int): Number of training epochs
            verbose (bool): Whether to print training progress
            
        Returns:
            list: Training loss history
        """
        loss_history = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            
            for i in indices:
                # Prepare data
                x = X_train[i].reshape(-1, 1)
                y = Y_train[i].reshape(-1, 1)
                
                # Forward pass
                y_hat, cache = self.forward(x)
                loss = self.cross_entropy_loss(y_hat, y)
                total_loss += loss
                
                # Backward pass
                dW1, db1, dW2, db2 = self.backward(y, cache)
                
                # Update parameters
                self.update_parameters(dW1, db1, dW2, db2)
            
            # Record average loss
            avg_loss = total_loss / len(X_train)
            loss_history.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}, Loss: {avg_loss:.4f}")
        
        return loss_history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Predicted class labels
        """
        predictions = []
        
        for i in range(len(X)):
            x = X[i].reshape(-1, 1)
            y_hat, _ = self.forward(x)
            pred = np.argmax(y_hat)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def evaluate(self, X, y):
        """
        Evaluate the model accuracy.
        
        Args:
            X (np.array): Input features
            y (np.array): True class labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return correct / len(y)
