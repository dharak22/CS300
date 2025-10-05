# CS300 - Custom SeparableConv2D Implementation

## Project Overview

This repository contains a custom implementation of TensorFlow/Keras SeparableConv2D layer with enhanced features designed to improve model performance and flexibility. The project includes comprehensive testing and comparison against the standard SeparableConv2D implementation on the MNIST dataset.

## Features

The `CustomSeparableConv2D` layer extends the standard Keras SeparableConv2D with several advanced capabilities:

### Enhanced Features
- **Separate Activation Functions**: Independent activation functions for depthwise and pointwise convolutions
- **Depthwise Scaling**: Configurable scaling factor for depthwise convolution outputs
- **Skip Connections**: Automatic residual connections with learnable projection when needed
- **Weight Normalization**: Optional L2 normalization of convolutional kernels for training stability
- **Flexible Configuration**: Fully compatible with standard Keras layer API

## Project Structure

```
CS300/
├── README.md                              #Project documentation
├── custom_con2d.py                        #Core CustomSeparableConv2D implementation
├── testing_custom_conv2d.py               #Comprehensive testing and comparison script
└── separable_conv_comparison_results/     #Experimental results and visualizations
    ├── standard_model/                    #Standard SeparableConv2D results
        └── classification_report.json
        ├── confusion_matrix.png
        ├── sample_predictions.png
        ├── training_log.csv
                            
    ├── custom_model/                      #CustomSeparableConv2D results
        └── classification_report.json
        ├── confusion_matrix.png
        ├── sample_predictions.png
        ├── training_log.csv               
```

## Installation

### Requirements
```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
```

### Quick Start
```python
from custom_con2d import CustomSeparableConv2D

# Basic usage (drop-in replacement)
layer = CustomSeparableConv2D(filters=64, kernel_size=3)

# Advanced usage with custom features
layer = CustomSeparableConv2D(
    filters=64,
    kernel_size=3,
    padding='same',
    depthwise_activation='relu',
    pointwise_activation='relu',
    normalize_weights=True,
    add_skip_connection=True,
    depthwise_scale=1.2
)
```

## Experimental Setup

### Dataset
- **MNIST**: 28×28 grayscale handwritten digit images
- **Training**: 55,000 samples
- **Validation**: 5,000 samples  
- **Testing**: 10,000 samples

### Model Architecture
Both standard and custom models use identical architectures:
- 3 Convolutional blocks with batch normalization and max pooling
- Global average pooling
- Dense classification head with dropout (0.5)
- Total parameters: ~100K

### Training Configuration
- **Optimizer**: Adam (learning rate: 0.001)
- **Batch Size**: 128
- **Epochs**: 20 (with early stopping)
- **Loss**: Categorical crossentropy

## Results

Detailed experimental results are available in the `separable_conv_comparison_results/` folder:

### Performance Metrics
- **Accuracy Comparison**: Validation and test accuracy curves for both models
- **Training Efficiency**: Training time and convergence speed analysis
- **Classification Reports**: Per-class precision, recall, and F1-scores (JSON format)

### Visualizations
- **Confusion Matrices**: Detailed error analysis for both models
- **Sample Predictions**: Visual comparison of correct/incorrect predictions
- **Training Curves**: Loss and accuracy progression over epochs
- **Model Architectures**: Complete layer-by-layer diagrams

### Key Findings
Results demonstrate the impact of custom features on:
- Model accuracy and generalization
- Training stability and convergence
- Feature representation quality

*See `separable_conv_comparison_results/metrics_comparison.json` for quantitative results and `experiment_log.txt` for detailed training logs.*

## Usage Examples

### Basic Model Building
```python
from tensorflow import keras
from custom_con2d import CustomSeparableConv2D

model = keras.Sequential([
    CustomSeparableConv2D(32, 3, padding='same', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(2),
    # ... additional layers
])
```

### Running Experiments
```bash
# Run full comparison experiment
python testing_custom_conv2d.py

# Results will be saved to separable_conv_comparison_results/
```

## Custom Features Explanation

### Depthwise & Pointwise Activations
Apply different non-linearities at each stage of the separable convolution:
```python
CustomSeparableConv2D(
    filters=64,
    kernel_size=3,
    depthwise_activation='relu',  # After depthwise conv
    pointwise_activation='swish'   # After pointwise conv
)
```

### Weight Normalization
Normalizes kernels to unit length for improved gradient flow:
```python
CustomSeparableConv2D(64, 3, normalize_weights=True)
```

### Skip Connections
Enables residual learning with automatic projection:
```python
CustomSeparableConv2D(64, 3, add_skip_connection=True)
```

## Technical Details

### Implementation Highlights
- **Gradient-Safe Operations**: All custom operations support backpropagation
- **Keras Compatible**: Full integration with model serialization and callbacks
- **Memory Efficient**: Minimal overhead compared to standard implementation

### Compatibility
- TensorFlow 2.x
- Keras 3.x
- Python 3.8+

## Contributing

Contributions are welcome! Areas for improvement:
- Additional custom features (channel attention, dynamic kernels)
- Performance optimization
- Extended benchmark datasets (CIFAR-10, ImageNet)


## Acknowledgments

- Built on TensorFlow/Keras framework
- Inspired by modern CNN architecture innovations (MobileNet, EfficientNet)
- MNIST dataset from Yann LeCun's database

---

