import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.src import ops
from keras.src import initializers
from keras.src import regularizers
from keras.src import constraints
from keras.src import activations
from keras.src.layers import SeparableConv2D, Input, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D
from keras.src.models import Model
from keras.src.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import time
import os
import json
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# At the beginning of your script
RESULTS_DIR = "separable_conv_comparison_results"
os.makedirs(RESULTS_DIR, exist_ok=True)  # Main results dir
os.makedirs(os.path.join(RESULTS_DIR, "standard_model"), exist_ok=True)  # Subdirs
os.makedirs(os.path.join(RESULTS_DIR, "custom_model"), exist_ok=True)


class CustomSeparableConv2D(SeparableConv2D):
    
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        depth_multiplier=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        depthwise_activation=None,
        pointwise_activation=None,
        depthwise_scale=1.0,
        add_skip_connection=False,
        normalize_weights=False,
        **kwargs
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=depth_multiplier,
            activation=None,  # Handled separately
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            pointwise_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            pointwise_regularizer=pointwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            pointwise_constraint=pointwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        
        self.depthwise_activation = activations.get(depthwise_activation)
        self.pointwise_activation = activations.get(pointwise_activation or activation)
        self.depthwise_scale = depthwise_scale
        self.add_skip_connection = add_skip_connection
        self.normalize_weights = normalize_weights
        self.skip_projection = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # Initialize skip projection weights if needed
        if self.add_skip_connection:
            input_channels = input_shape[-1] if self.data_format == 'channels_last' else input_shape[1]
            self.skip_projection = self.add_weight(
                name='skip_projection',
                shape=(1, 1, input_channels, self.filters),
                initializer=self.pointwise_initializer,
                regularizer=self.pointwise_regularizer,
                constraint=self.pointwise_constraint,
                trainable=True
            )

    def _normalize_weights(self, weights):
        """Safe weight normalization with gradient support."""
        square_sum = ops.sum(ops.square(weights), axis=[0, 1, 2], keepdims=True)
        inv_norm = ops.rsqrt(ops.maximum(square_sum, 1e-7))
        return ops.multiply(weights, inv_norm)

def call(self, inputs):
    # Optional weight normalization
    depthwise_kernel = (self._normalize_weights(self.depthwise_kernel) 
                      if self.normalize_weights else self.depthwise_kernel)
    pointwise_kernel = (self._normalize_weights(self.pointwise_kernel) 
                       if self.normalize_weights else self.pointwise_kernel)

    # Depthwise convolution
    if hasattr(ops, 'depthwise_conv2d'):  # Newer Keras versions
        x = ops.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
    else:  # Older versions
        x = ops.depthwise_conv(
            inputs,
            depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )
    
    # Depthwise processing
    if self.depthwise_activation:
        x = self.depthwise_activation(x)
    if self.depthwise_scale != 1.0:
        x = x * self.depthwise_scale

    # Pointwise convolution 
    if hasattr(ops, 'conv'):
        outputs = ops.conv(
            x,
            pointwise_kernel,
            strides=1,
            padding="valid",
            data_format=self.data_format,
            dilation_rate=1,
        )
    else:
        
        outputs = tf.nn.conv2d(
            x,
            pointwise_kernel,
            strides=1,
            padding="VALID",
            data_format='NHWC' if self.data_format == 'channels_last' else 'NCHW',
        )
    
    # Rest of your call method remains the same...
    if self.use_bias:
        outputs = outputs + ops.reshape(
            self.bias,
            (1, 1, 1, self.filters) if self.data_format == 'channels_last' 
            else (1, self.filters, 1, 1)
        )

    if self.add_skip_connection:
        if inputs.shape == outputs.shape:
            outputs = outputs + inputs
        else:
            outputs = outputs + ops.conv(
                inputs,
                self.skip_projection,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

    if self.pointwise_activation:
        outputs = self.pointwise_activation(outputs)
        
    return outputs

def get_config(self):
    config = super().get_config()
    config.update({
        'depthwise_activation': activations.serialize(self.depthwise_activation),
        'pointwise_activation': activations.serialize(self.pointwise_activation),
        'depthwise_scale': self.depthwise_scale,
        'add_skip_connection': self.add_skip_connection,
        'normalize_weights': self.normalize_weights,
    })
    return config

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_file = os.path.join(log_dir, "experiment_log.txt")
        
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")

logger = ExperimentLogger(RESULTS_DIR)

def load_and_preprocess_mnist():
    logger.log("Loading and preprocessing MNIST dataset")
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Reshape and normalize
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
    
    # One-hot encode
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # Validation split
    val_split = 5000
    x_val, y_val = x_train[-val_split:], y_train[-val_split:]
    x_train, y_train = x_train[:-val_split], y_train[:-val_split]
    
    logger.log(f"Dataset shapes - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def build_model(base_conv_layer, name):
    inputs = Input(shape=(28, 28, 1))
    
    # Conv Block 1
    x = base_conv_layer(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)
    
    # Conv Block 2
    x = base_conv_layer(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)
    
    # Conv Block 3 with custom features
    conv_args = {}
    if base_conv_layer == CustomSeparableConv2D:
        conv_args.update({
            'depthwise_activation': 'relu',
            'pointwise_activation': 'relu',
            'normalize_weights': True,
            'add_skip_connection': True
        })
    
    x = base_conv_layer(128, 3, padding='same', **conv_args)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    return Model(inputs, outputs, name=name)

def train_model(model, train_data, val_data, model_name):
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # 1. Create model directory FIRST
    model_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. Setup callbacks AFTER directory exists
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        CSVLogger(os.path.join(model_dir, 'training_log.csv')),  # Now safe
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # 3. Compile and train
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=20,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2
    )
    
    return history.history, time.time() - start_time

def evaluate_model(model, test_data, model_name):
    x_test, y_test = test_data
    
    # Create model directory
    model_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Test evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.log(f"{model_name} - Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Save classification report
    report = classification_report(
        y_true_classes, y_pred_classes,
        target_names=[str(i) for i in range(10)],
        output_dict=True
    )
    with open(os.path.join(model_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Example predictions
    plot_sample_predictions(model, x_test[:30], y_test[:30], model_dir)
    
    return test_acc

def plot_sample_predictions(model, x_samples, y_true, save_dir):
    y_pred = model.predict(x_samples)
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    plt.figure(figsize=(15, 10))
    for i in range(min(30, len(x_samples))):
        plt.subplot(5, 6, i+1)
        plt.imshow(x_samples[i].squeeze(), cmap='gray')
        color = 'green' if y_pred_classes[i] == y_true_classes[i] else 'red'
        plt.title(f"T:{y_true_classes[i]}\nP:{y_pred_classes[i]}", color=color)
        plt.axis('off')
    
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'))
    plt.close()

def compare_results(standard_res, custom_res):
    # Accuracy/Loss Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(standard_res['history']['val_accuracy'], label='Standard Val')
    plt.plot(custom_res['history']['val_accuracy'], label='Custom Val')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(standard_res['history']['val_loss'], label='Standard Val')
    plt.plot(custom_res['history']['val_loss'], label='Custom Val')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_comparison.png'))
    plt.close()
    
    # Metrics Comparison
    metrics = {
        'Model': ['Standard', 'Custom'],
        'Test Accuracy': [standard_res['test_acc'], custom_res['test_acc']],
        'Training Time (s)': [standard_res['time'], custom_res['time']]
    }
    
    with open(os.path.join(RESULTS_DIR, 'metrics_comparison.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    try:
        logger.log("Starting MNIST SeparableConv2D Comparison Experiment")
        
        # Load data
        train_data, val_data, test_data = load_and_preprocess_mnist()
        
        # Build and train standard model
        standard_model = build_model(SeparableConv2D, 'standard_model')
        standard_model.summary()
        keras.utils.plot_model(
            standard_model,
            to_file=os.path.join(RESULTS_DIR, 'standard_model_arch.png'),
            show_shapes=True
        )
        
        std_history, std_time = train_model(standard_model, train_data, val_data, 'standard_model')
        std_acc = evaluate_model(standard_model, test_data, 'standard_model')
        
        # Build and train custom model
        custom_model = build_model(CustomSeparableConv2D, 'custom_model')
        custom_model.summary()
        keras.utils.plot_model(
            custom_model,
            to_file=os.path.join(RESULTS_DIR, 'custom_model_arch.png'),
            show_shapes=True
        )
        
        custom_history, custom_time = train_model(custom_model, train_data, val_data, 'custom_model')
        custom_acc = evaluate_model(custom_model, test_data, 'custom_model')
        
        # Compare results
        compare_results(
            {'history': std_history, 'test_acc': std_acc, 'time': std_time},
            {'history': custom_history, 'test_acc': custom_acc, 'time': custom_time}
        )
        
        # Final report
        improvement = custom_acc - std_acc
        logger.log(f"\nResults Summary:")
        logger.log(f"Standard Model Accuracy: {std_acc:.4f}")
        logger.log(f"Custom Model Accuracy: {custom_acc:.4f}")
        logger.log(f"Accuracy Improvement: {improvement:.4f} ({improvement/std_acc*100:.2f}%)")
        logger.log(f"\nTraining Times:")
        logger.log(f"Standard: {std_time:.2f}s")
        logger.log(f"Custom: {custom_time:.2f}s")
        
    except Exception as e:
        logger.log(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()