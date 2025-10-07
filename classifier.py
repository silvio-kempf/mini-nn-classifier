#!/usr/bin/env python3
"""
Digit Classifier - Main Script

A simple command-line tool to classify handwritten digits using a pre-trained neural network.

Usage:
    python classifier.py image.png
    python classifier.py --create-sample 5
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from neural_network import MultiClassNeuralNetwork


def load_pretrained_model():
    """Load the pre-trained model."""
    model_path = "trained_model.npy"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pre-trained model not found at {model_path}. "
            "Please ensure the trained_model.npy file is in the same directory."
        )
    
    print("Loading pre-trained model...")
    try:
        # Load model parameters
        params = np.load(model_path, allow_pickle=True).item()
        model = MultiClassNeuralNetwork(
            n_input=64,
            n_hidden=32,
            n_output=10,
            learning_rate=0.1,
            seed=42
        )
        model.W1 = params['W1']
        model.b1 = params['b1']
        model.W2 = params['W2']
        model.b2 = params['b2']
        print("Model loaded successfully!")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def preprocess_image(image_path_or_array, target_size=(8, 8), debug=False):
    """Preprocess an image for digit classification."""
    # Load image
    if isinstance(image_path_or_array, str):
        image = Image.open(image_path_or_array)
    elif isinstance(image_path_or_array, np.ndarray):
        if image_path_or_array.shape[-1] == 3:  # RGB
            image = Image.fromarray(image_path_or_array.astype(np.uint8))
        else:  # Grayscale
            image = Image.fromarray(image_path_or_array.astype(np.uint8), mode='L')
    else:
        image = image_path_or_array
    
    if debug:
        print(f"Original image size: {image.size}")
        print(f"Original image mode: {image.mode}")
    
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array for processing
    img_array = np.array(image, dtype=np.float32)
    
    if debug:
        print(f"Original pixel range: [{img_array.min():.1f}, {img_array.max():.1f}]")
        print(f"Original mean: {img_array.mean():.1f}")
    
    # Step 1: Find the digit region (crop to content)
    img_array = crop_to_content(img_array)
    
    if debug:
        print(f"After cropping: {img_array.shape}")
    
    # Step 2: Resize to target size with better interpolation
    intermediate_size = (target_size[0] * 4, target_size[1] * 4)
    temp_image = Image.fromarray(img_array.astype(np.uint8))
    temp_image = temp_image.resize(intermediate_size, Image.Resampling.LANCZOS)
    temp_image = temp_image.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(temp_image, dtype=np.float32)
    
    # Step 3: Normalize contrast and brightness
    img_array = normalize_contrast(img_array)
    
    # Step 4: Handle inversion (MNIST digits are white on black)
    img_array = handle_inversion(img_array)
    
    # Step 5: Final normalization to [0, 1] range
    img_array = img_array / 255.0
    
    if debug:
        print(f"Final pixel range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        print(f"Final mean: {img_array.mean():.3f}")
    
    return img_array.flatten()


def crop_to_content(img_array, padding=2):
    """Crop image to the content area (remove empty borders)."""
    # Find non-zero (content) pixels
    rows = np.any(img_array < 240, axis=1)  # Assume background is light
    cols = np.any(img_array < 240, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return img_array
    
    # Get bounding box
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add padding
    rmin = max(0, rmin - padding)
    rmax = min(img_array.shape[0], rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(img_array.shape[1], cmax + padding)
    
    return img_array[rmin:rmax, cmin:cmax]


def normalize_contrast(img_array):
    """Normalize contrast to improve digit visibility."""
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    
    if max_val > min_val:
        img_array = (img_array - min_val) / (max_val - min_val) * 255
    else:
        img_array = np.zeros_like(img_array)
    
    return img_array


def handle_inversion(img_array):
    """Handle image inversion to match MNIST format (white digits on black background)."""
    mean_brightness = np.mean(img_array)
    
    if mean_brightness > 128:
        img_array = 255 - img_array
    
    return img_array


def classify_image(model, image_path_or_array, show_image=True, debug=False):
    """Classify a single image using the trained model."""
    # Preprocess the image
    processed_image = preprocess_image(image_path_or_array, debug=debug)
    
    # Make prediction
    x = processed_image.reshape(-1, 1)
    y_hat, _ = model.forward(x)
    
    # Get prediction and confidence
    predicted_class = np.argmax(y_hat)
    confidence = float(np.max(y_hat))
    probabilities = y_hat.flatten()
    
    if show_image:
        # Display the image and prediction
        plt.figure(figsize=(8, 4))
        
        # Show original processed image
        plt.subplot(1, 2, 1)
        plt.imshow(processed_image.reshape(8, 8), cmap='gray')
        plt.title(f'Input Image\nPredicted: {predicted_class}\nConfidence: {confidence:.3f}')
        plt.axis('off')
        
        # Show probability distribution
        plt.subplot(1, 2, 2)
        plt.bar(range(10), probabilities)
        plt.title('Class Probabilities')
        plt.xlabel('Digit Class')
        plt.ylabel('Probability')
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)
        
        # Highlight the predicted class
        plt.bar(predicted_class, confidence, color='red', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    return predicted_class, confidence, probabilities


def main():
    """Main function to classify images using the pre-trained model."""
    parser = argparse.ArgumentParser(
        description='Classify handwritten digits using a pre-trained neural network',
        epilog='Example: python classifier.py my_digit.png'
    )
    parser.add_argument('image', nargs='?', type=str, help='Path to image file to classify')
    parser.add_argument('--no-display', action='store_true', help='Do not show image visualization')
    parser.add_argument('--debug', action='store_true', help='Show detailed preprocessing information')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nThe model is pre-trained and ready to classify your handwritten digits!")
        print("Just provide an image file as an argument.")
        print("\nSupported image formats: PNG, JPG, JPEG, BMP, TIFF")
        print("The image will be automatically resized and preprocessed.")
        return
    
    
    # Classify an image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found!")
            return
        
        print(f"Loading and classifying image: {args.image}")
        model = load_pretrained_model()
        pred, conf, probs = classify_image(model, args.image, show_image=not args.no_display, debug=args.debug)
        
        print(f"\nClassification Results:")
        print(f"Predicted Digit: {pred}")
        print(f"Confidence: {conf:.3f}")
        print(f"All Probabilities: {[f'{p:.3f}' for p in probs]}")
        return
    
    # If we get here, show help
    parser.print_help()


if __name__ == "__main__":
    main()
