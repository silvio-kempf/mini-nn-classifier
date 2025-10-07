# Mini Neural Network - Digit Classifier

A simple neural network for handwritten digit classification built from scratch using only NumPy.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test with Provided Images
```bash
# Test with pre-made images (0-9)
python classifier.py test_images/digit_5.png
python classifier.py test_images/digit_7.png
```

### 3. Test Your Own Images
```bash
# Test your own handwritten digit
python classifier.py my_digit.png
```

## 📁 What's Included

- **10 test images** in `test_images/` folder (digits 0-9)
  - **Guaranteed**: These images were NOT used during training
  - **Source**: Real MNIST test samples from the same test set used for evaluation
- **Pre-trained model** with 98%+ accuracy
- **Simple CLI** for image classification
- **Clean code structure** - only essential files

## 🎯 Usage Examples

```bash
# Test a specific digit
python classifier.py test_images/digit_3.png

# Test your own image
python classifier.py my_handwritten_5.png

# Debug mode (see preprocessing details)
python classifier.py my_image.png --debug

# No visualization
python classifier.py my_image.png --no-display
```

## 📊 Model Performance

- **Training Accuracy**: ~100%
- **Test Accuracy**: ~98%
- **Architecture**: 64 → 32 → 10 neurons
- **Dataset**: MNIST handwritten digits (8x8 pixels)

## ⚠️ Important: Model Limitations

**This model was trained specifically on MNIST handwritten digits and works best with:**
- **Simple handwritten digits** (0-9)
- **Clear, single digits** on light backgrounds
- **Good contrast** between digit and background
- **No complex graphics or text**

**The model may not work well with:**
- Complex images with multiple elements
- Printed text or fonts
- Low contrast images
- Images with backgrounds or noise


## 🛠️ For Developers & Transparency

```bash
# Retrain the model (shows how it was created)
python train_model.py --epochs 100

# Experiment with different parameters
python train_model.py --epochs 50 --hidden 64 --lr 0.05
```

## 📝 Image Requirements

- **Formats**: PNG, JPG, JPEG, BMP, TIFF
- **Size**: Any (automatically resized to 8x8)
- **Best results**: Simple handwritten digits like those in MNIST dataset
- **Content**: Single digit (0-9) on light background
- **Quality**: Clear, high contrast, minimal noise

---

**Perfect for learning neural networks from scratch!** 🎓