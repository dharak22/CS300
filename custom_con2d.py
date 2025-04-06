import tensorflow as tf
from keras.src.layers import SeparableConv2D
from keras.src import ops
from keras.src import initializers
from keras.src import regularizers
from keras.src import constraints
from keras.src import activations
    
class CustomSeparableConv2D(SeparableConv2D):
    """Custom 2D separable convolution layer with enhanced features."""
    
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
        outputs = ops.conv2d(
            x,
            pointwise_kernel,
            strides=1,
            padding="valid",
            data_format=self.data_format,
            dilation_rate=1,
        )
        
        # Bias addition
        if self.use_bias:
            outputs = outputs + ops.reshape(
                self.bias,
                (1, 1, 1, self.filters) if self.data_format == 'channels_last' 
                else (1, self.filters, 1, 1)
            )

        # Skip connection handling
        if self.add_skip_connection:
            if inputs.shape == outputs.shape:
                outputs = outputs + inputs
            else:
                # Use pre-built projection weights
                outputs = outputs + ops.conv2d(
                    inputs,
                    self.skip_projection,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )

        # Final activation
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