"""
Model architecture module for ApneaGuard.
Defines CNN and CNN-LSTM models for sleep apnea detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import config


def build_cnn_model(input_shape: tuple) -> keras.Model:
    """
    Build 1D Convolutional Neural Network for ECG classification.
    
    Architecture:
    - Multiple 1D convolutional layers to extract temporal patterns
    - Batch normalization for stable training
    - Max pooling to reduce dimensionality
    - Dropout for regularization
    - Dense layers for classification
    
    The CNN learns to recognize patterns in the raw ECG signal that are
    indicative of apnea events (e.g., changes in heart rate patterns).
    
    Args:
        input_shape: Shape of input data (window_samples, 1)
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='CNN_ApneaDetector')
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # Convolutional Block 1
    model.add(layers.Conv1D(
        filters=config.CNN_FILTERS[0],
        kernel_size=config.CNN_KERNEL_SIZE,
        activation='relu',
        padding='same',
        name='conv1'
    ))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.MaxPooling1D(pool_size=config.CNN_POOL_SIZE, name='pool1'))
    model.add(layers.Dropout(0.3, name='dropout1'))
    
    # Convolutional Block 2
    model.add(layers.Conv1D(
        filters=config.CNN_FILTERS[1],
        kernel_size=config.CNN_KERNEL_SIZE,
        activation='relu',
        padding='same',
        name='conv2'
    ))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.MaxPooling1D(pool_size=config.CNN_POOL_SIZE, name='pool2'))
    model.add(layers.Dropout(0.4, name='dropout2'))
    
    # Convolutional Block 3
    model.add(layers.Conv1D(
        filters=config.CNN_FILTERS[2],
        kernel_size=config.CNN_KERNEL_SIZE,
        activation='relu',
        padding='same',
        name='conv3'
    ))
    model.add(layers.BatchNormalization(name='bn3'))
    model.add(layers.MaxPooling1D(pool_size=config.CNN_POOL_SIZE, name='pool3'))
    model.add(layers.Dropout(0.5, name='dropout3'))
    
    # Flatten to 1D for dense layers
    model.add(layers.Flatten(name='flatten'))
    
    # Dense Block 1
    model.add(layers.Dense(
        config.DENSE_UNITS[0],
        activation='relu',
        name='dense1'
    ))
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout4'))
    
    # Dense Block 2
    model.add(layers.Dense(
        config.DENSE_UNITS[1],
        activation='relu',
        name='dense2'
    ))
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout5'))
    
    # Output layer - Binary classification (apnea vs normal)
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    return model


def build_cnn_lstm_model(input_shape: tuple) -> keras.Model:
    """
    Build CNN-LSTM hybrid model for ECG classification.
    
    Architecture:
    - CNN layers: Extract local temporal features from ECG
    - LSTM layers: Capture long-term dependencies and temporal patterns
    - Dense layers: Final classification
    
    The combination of CNN and LSTM allows the model to:
    - Learn hierarchical features (CNN)
    - Understand temporal sequences and context (LSTM)
    
    This is particularly useful for sleep apnea detection because:
    - Apnea events have characteristic patterns over time
    - Heart rate changes evolve gradually during apnea episodes
    
    Args:
        input_shape: Shape of input data (window_samples, 1)
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='CNN_LSTM_ApneaDetector')
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # CNN Block 1
    model.add(layers.Conv1D(
        filters=config.CNN_FILTERS[0],
        kernel_size=config.CNN_KERNEL_SIZE,
        activation='relu',
        padding='same',
        name='conv1'
    ))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.MaxPooling1D(pool_size=config.CNN_POOL_SIZE, name='pool1'))
    
    # CNN Block 2
    model.add(layers.Conv1D(
        filters=config.CNN_FILTERS[1],
        kernel_size=config.CNN_KERNEL_SIZE,
        activation='relu',
        padding='same',
        name='conv2'
    ))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.MaxPooling1D(pool_size=config.CNN_POOL_SIZE, name='pool2'))
    
    # LSTM Block 1 - Return sequences for stacked LSTM
    model.add(layers.LSTM(
        config.LSTM_UNITS[0],
        return_sequences=True,
        name='lstm1'
    ))
    model.add(layers.Dropout(0.4, name='dropout1'))
    
    # LSTM Block 2 - Final LSTM layer
    model.add(layers.LSTM(
        config.LSTM_UNITS[1],
        return_sequences=False,
        name='lstm2'
    ))
    model.add(layers.Dropout(0.5, name='dropout2'))
    
    # Dense Block
    model.add(layers.Dense(
        config.DENSE_UNITS[0],
        activation='relu',
        name='dense1'
    ))
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout3'))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    return model


def compile_model(model: keras.Model, learning_rate: float = None) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer (None = use config)
        
    Returns:
        Compiled model
    """
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    # Select optimizer
    if config.OPTIMIZER == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif config.OPTIMIZER == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif config.OPTIMIZER == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=config.LOSS_FUNCTION,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_model(model_type: str, input_shape: tuple) -> keras.Model:
    """
    Create and compile model based on specified type.
    
    Args:
        model_type: Type of model ('cnn' or 'cnn_lstm')
        input_shape: Shape of input data
        
    Returns:
        Compiled Keras model
    """
    print(f"\nBuilding {model_type.upper()} model...")
    
    # Build model architecture
    if model_type == 'cnn':
        model = build_cnn_model(input_shape)
    elif model_type == 'cnn_lstm':
        model = build_cnn_lstm_model(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compile model
    model = compile_model(model)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    return model


if __name__ == "__main__":
    """
    Test model creation.
    """
    print("Testing model architectures...")
    
    # Calculate input shape
    # For 60-second window at 100 Hz: 60 * 100 = 6000 samples
    window_samples = config.WINDOW_SIZE * config.SAMPLING_RATE
    input_shape = (window_samples, 1)
    
    print(f"\nInput shape: {input_shape}")
    
    # Test CNN model
    print("\n" + "="*50)
    print("Testing CNN Model")
    print("="*50)
    cnn_model = create_model('cnn', input_shape)
    
    # Test CNN-LSTM model
    print("\n" + "="*50)
    print("Testing CNN-LSTM Model")
    print("="*50)
    cnn_lstm_model = create_model('cnn_lstm', input_shape)
    
    print("\nModel creation test complete!")
