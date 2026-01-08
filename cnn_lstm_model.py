"""
CNN-LSTM Model Architecture Module
===================================
Defines the hybrid CNN-LSTM deep learning model for electrical fault classification.

Architecture:
    Input (timesteps, features)
        ↓
    CNN Feature Extraction (Conv1D + MaxPooling + Dropout)
        ↓
    LSTM Temporal Learning
        ↓
    Dense Classification (Softmax)
        ↓
    Output (fault class probabilities)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_cnn_lstm_model(input_shape, num_classes, config=None):
    """
    Build hybrid CNN-LSTM model for time-series fault classification.
    
    Architecture:
        1. CNN layers: Extract spatial/local patterns from time-series
        2. LSTM layer: Learn temporal dependencies across sequences
        3. Dense layers: Final classification
    
    Args:
        input_shape (tuple): (timesteps, features) e.g., (10, 6)
        num_classes (int): Number of fault types (typically 6)
        config (dict): Optional configuration for hyperparameters
            - cnn_filters_1: First Conv1D filters (default: 64)
            - cnn_filters_2: Second Conv1D filters (default: 128)
            - cnn_kernel_size: Kernel size for Conv1D (default: 3)
            - lstm_units: LSTM units (default: 100)
            - dropout_rate: Dropout rate (default: 0.3)
            - dense_units: Dense layer units (default: 64)
    
    Returns:
        keras.Model: Compiled CNN-LSTM model
    """
    # Default configuration
    default_config = {
        'cnn_filters_1': 64,
        'cnn_filters_2': 128,
        'cnn_kernel_size': 3,
        'lstm_units': 100,
        'dropout_rate': 0.3,
        'dense_units': 64,
        'learning_rate': 0.001
    }
    
    if config is not None:
        default_config.update(config)
    
    cfg = default_config
    
    # Build model
    model = Sequential([
        # ===== CNN Feature Extraction Block 1 =====
        Conv1D(
            filters=cfg['cnn_filters_1'],
            kernel_size=cfg['cnn_kernel_size'],
            activation='relu',
            padding='same',
            input_shape=input_shape,
            name='conv1d_1'
        ),
        BatchNormalization(name='batch_norm_1'),
        MaxPooling1D(pool_size=2, name='maxpool_1'),
        Dropout(cfg['dropout_rate'], name='dropout_1'),
        
        # ===== CNN Feature Extraction Block 2 =====
        Conv1D(
            filters=cfg['cnn_filters_2'],
            kernel_size=cfg['cnn_kernel_size'],
            activation='relu',
            padding='same',
            name='conv1d_2'
        ),
        BatchNormalization(name='batch_norm_2'),
        MaxPooling1D(pool_size=2, name='maxpool_2'),
        Dropout(cfg['dropout_rate'], name='dropout_2'),
        
        # ===== LSTM Temporal Learning =====
        LSTM(
            units=cfg['lstm_units'],
            return_sequences=False,  # Only return last output
            dropout=cfg['dropout_rate'],
            recurrent_dropout=0.2,
            name='lstm'
        ),
        
        # ===== Dense Classification Layers =====
        Dense(cfg['dense_units'], activation='relu', name='dense_1'),
        Dropout(cfg['dropout_rate'] + 0.1, name='dropout_3'),  # Slightly higher dropout
        Dense(num_classes, activation='softmax', name='output')
    ], name='CNN_LSTM_FaultClassifier')
    
    return model


def compile_model(model, learning_rate=0.001, metrics=None):
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        metrics: List of metrics (default: accuracy, precision, recall)
    
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    
    return model


def get_callbacks(model_save_path='best_cnn_lstm_model.h5', patience=10):
    """
    Create training callbacks for monitoring and optimization.
    
    Args:
        model_save_path: Path to save best model
        patience: Patience for early stopping
    
    Returns:
        list: Keras callbacks
    """
    callbacks = [
        # Early stopping: stop if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Model checkpoint: save best model
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
    ]
    
    return callbacks


def print_model_summary(model, save_to_file=None):
    """
    Print detailed model summary.
    
    Args:
        model: Keras model
        save_to_file: Optional path to save summary as text
    """
    print("=" * 80)
    print("CNN-LSTM Model Architecture Summary")
    print("=" * 80)
    
    model.summary()
    
    print("\n" + "=" * 80)
    print("Model Configuration")
    print("=" * 80)
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print("=" * 80)
    
    if save_to_file:
        with open(save_to_file, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"✓ Model summary saved to {save_to_file}")


def build_lightweight_model(input_shape, num_classes):
    """
    Build a lighter CNN-LSTM model for faster training (alternative architecture).
    
    Args:
        input_shape (tuple): (timesteps, features)
        num_classes (int): Number of output classes
    
    Returns:
        keras.Model: Compiled lightweight model
    """
    model = Sequential([
        # Single CNN block
        Conv1D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.3),
        
        # LSTM
        LSTM(50, dropout=0.3, recurrent_dropout=0.2),
        
        # Output
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ], name='Lightweight_CNN_LSTM')
    
    return model


def build_deep_model(input_shape, num_classes):
    """
    Build a deeper CNN-LSTM model for potentially better performance (alternative architecture).
    
    Args:
        input_shape (tuple): (timesteps, features)
        num_classes (int): Number of output classes
    
    Returns:
        keras.Model: Compiled deep model
    """
    model = Sequential([
        # CNN Block 1
        Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),
        
        # CNN Block 2
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),
        
        # CNN Block 3
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layers
        LSTM(150, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        LSTM(100, dropout=0.3, recurrent_dropout=0.2),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ], name='Deep_CNN_LSTM')
    
    return model


# Testing and demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("CNN-LSTM Model Architecture Module - Test Run")
    print("=" * 80)
    
    # Test model building
    INPUT_SHAPE = (10, 6)  # 10 timesteps, 6 features
    NUM_CLASSES = 6
    
    print("\n1. Building standard CNN-LSTM model...")
    model = build_cnn_lstm_model(INPUT_SHAPE, NUM_CLASSES)
    model = compile_model(model)
    print_model_summary(model)
    
    print("\n2. Building lightweight model...")
    light_model = build_lightweight_model(INPUT_SHAPE, NUM_CLASSES)
    light_model = compile_model(light_model)
    print(f"Lightweight model parameters: {light_model.count_params():,}")
    
    print("\n3. Building deep model...")
    deep_model = build_deep_model(INPUT_SHAPE, NUM_CLASSES)
    deep_model = compile_model(deep_model)
    print(f"Deep model parameters: {deep_model.count_params():,}")
    
    print("\n4. Testing callbacks...")
    callbacks = get_callbacks()
    print(f"✓ Created {len(callbacks)} callbacks:")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}")
    
    print("\n5. Testing with dummy data...")
    import numpy as np
    X_dummy = np.random.randn(100, 10, 6)
    y_dummy = np.random.randint(0, 6, 100)
    
    print(f"  Input shape: {X_dummy.shape}")
    print(f"  Labels shape: {y_dummy.shape}")
    
    # Quick training test
    print("\n  Training for 2 epochs (test)...")
    history = model.fit(
        X_dummy, y_dummy,
        epochs=2,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    print(f"  ✓ Training completed")
    print(f"    Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"    Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Test prediction
    print("\n6. Testing prediction...")
    predictions = model.predict(X_dummy[:5], verbose=0)
    print(f"  Prediction shape: {predictions.shape}")
    print(f"  Sample prediction (probabilities):")
    print(f"    {predictions[0]}")
    print(f"  Predicted class: {np.argmax(predictions[0])}")
    
    print("\n" + "=" * 80)
    print("✓ All model architecture tests passed!")
    print("=" * 80)
