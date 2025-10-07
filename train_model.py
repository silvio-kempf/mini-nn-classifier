#!/usr/bin/env python3
"""
Training script for the Multi-Class Neural Network.

This script shows how the model was trained and allows developers
to retrain or experiment with different parameters.

For transparency and educational purposes.
"""

from neural_network import MultiClassNeuralNetwork
import argparse
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_and_prepare_data():
    """Load and prepare the digits dataset."""
    # Load data
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Normalize pixel values
    X = X / 16.0
    
    # Split into train/test
    np.random.seed(42)
    N = len(X)
    indices = np.random.permutation(N)
    split = int(0.8 * N)
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # One-hot encode labels
    num_classes = 10
    Y_train = np.eye(num_classes)[y_train]
    Y_test = np.eye(num_classes)[y_test]
    
    return X_train, X_test, y_train, y_test, Y_train, Y_test


def plot_training_history(loss_history):
    """Plot training loss history."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()


def visualize_predictions(X_test, y_test, model, n_samples=10):
    """Visualize model predictions on test samples."""
    # Get random samples
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        x = X_test[idx].reshape(-1, 1)
        y_true = y_test[idx]
        y_hat, _ = model.forward(x)
        y_pred = np.argmax(y_hat)
        
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap="gray")
        color = "green" if y_true == y_pred else "red"
        plt.title(f"True: {y_true}, Pred: {y_pred}", color=color)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def main():
    """Train the neural network model."""
    parser = argparse.ArgumentParser(
        description='Train the Multi-Class Neural Network (for transparency and education)',
        epilog='This script shows how the pre-trained model was created.'
    )
    parser.add_argument('--epochs', '-e', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Do not generate plots')
    parser.add_argument('--hidden', type=int, default=32,
                       help='Number of hidden neurons (default: 32)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    
    args = parser.parse_args()
    
    print("=== Training Multi-Class Neural Network ===")
    print("This script shows how the pre-trained model was created.")
    print("For transparency and educational purposes.\n")
    
    # Load and prepare data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, Y_train, Y_test = load_and_prepare_data()
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Initialize model
    print(f"\nInitializing neural network...")
    print(f"Architecture: {X_train.shape[1]} → {args.hidden} → 10")
    print(f"Learning rate: {args.lr}")
    
    model = MultiClassNeuralNetwork(
        n_input=64,
        n_hidden=args.hidden,
        n_output=10,
        learning_rate=args.lr,
        seed=42
    )
    
    # Train model
    print(f"\nTraining model for {args.epochs} epochs...")
    loss_history = model.train(X_train, Y_train, epochs=args.epochs, verbose=True)
    
    # Evaluate model
    print("\nEvaluating model...")
    train_accuracy = model.evaluate(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save model
    params = {
        'W1': model.W1,
        'b1': model.b1,
        'W2': model.W2,
        'b2': model.b2
    }
    np.save("trained_model.npy", params)
    print("\nModel saved as 'trained_model.npy'")
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        y_pred = model.predict(X_test)
        
        plot_training_history(loss_history)
        plot_confusion_matrix(y_test, y_pred, "Multi-Class Neural Network Confusion Matrix")
        visualize_predictions(X_test, y_test, model, n_samples=10)
    
    print("\nTraining complete!")
    print("You can now use the model with: python classifier.py your_image.png")
    
    return model, loss_history, test_accuracy


if __name__ == "__main__":
    main()
