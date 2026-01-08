"""
CNN-LSTM Training Script
=========================
Complete training pipeline for electrical fault classification.

Usage:
    python train_cnn_lstm.py
    
This script will:
    1. Load and preprocess data
    2. Create time-series sequences
    3. Build CNN-LSTM model
    4. Train with callbacks
    5. Save model and training history
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# Import custom modules
from cnn_lstm_preprocessing import (
    load_and_prepare_data,
    normalize_features,
    create_sequences,
    validate_sequence_shape,
    save_preprocessing_artifacts
)
from cnn_lstm_model import (
    build_cnn_lstm_model,
    compile_model,
    get_callbacks,
    print_model_summary
)

from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


# ============================================================================
# HYPERPARAMETERS
# ============================================================================
HYPERPARAMETERS = {
    # Data parameters
    'DATA_FILE': '../data/classData.csv',
    'TIME_STEPS': 10,
    'TEST_SIZE': 0.2,
    'VALIDATION_SPLIT': 0.2,
    'RANDOM_STATE': 42,
    
    # Model architecture
    'CNN_FILTERS_1': 64,
    'CNN_FILTERS_2': 128,
    'CNN_KERNEL_SIZE': 3,
    'LSTM_UNITS': 100,
    'DROPOUT_RATE': 0.3,
    'DENSE_UNITS': 64,
    
    # Training parameters
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'LEARNING_RATE': 0.001,
    'EARLY_STOPPING_PATIENCE': 10,
    
    # Output paths
    'MODEL_SAVE_PATH': '../models/best_cnn_lstm_model.h5',
    'HISTORY_SAVE_PATH': '../results/metrics/training_history.csv',
    'HYPERPARAMS_SAVE_PATH': '../results/metrics/hyperparameters_log.json'
}


def save_hyperparameters(hyperparams, filepath='hyperparameters_log.json'):
    """Save hyperparameters with timestamp."""
    log = {
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': hyperparams
    }
    
    with open(filepath, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"✓ Hyperparameters saved to {filepath}")


def save_training_history(history, filepath='training_history.csv'):
    """Save training history to CSV."""
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = range(1, len(history_df) + 1)
    history_df.to_csv(filepath, index=False)
    print(f"✓ Training history saved to {filepath}")


def plot_quick_summary(history, save_path='../results/visualizations/training_summary.png'):
    """Create a quick training summary plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training summary plot saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("CNN-LSTM ELECTRICAL FAULT CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ========================================================================
    # STEP 1: Load and Prepare Data
    # ========================================================================
    print("STEP 1: Loading and preparing data...")
    print("-" * 80)
    
    X, y, label_encoder = load_and_prepare_data(
        HYPERPARAMETERS['DATA_FILE'],
        verbose=True
    )
    
    # ========================================================================
    # STEP 2: Train-Test Split
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Splitting data into train and test sets...")
    print("-" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=HYPERPARAMETERS['TEST_SIZE'],
        stratify=y,
        random_state=HYPERPARAMETERS['RANDOM_STATE']
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set:  {X_test.shape[0]} samples")
    
    # ========================================================================
    # STEP 3: Normalize Features
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Normalizing features...")
    print("-" * 80)
    
    X_train_norm, X_test_norm, scaler = normalize_features(
        X_train, X_test,
        method='standard',
        verbose=True
    )
    
    # ========================================================================
    # STEP 4: Create Time-Series Sequences
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Creating time-series sequences...")
    print("-" * 80)
    
    TIME_STEPS = HYPERPARAMETERS['TIME_STEPS']
    
    X_train_seq, y_train_seq = create_sequences(
        X_train_norm, y_train,
        time_steps=TIME_STEPS,
        verbose=True
    )
    
    X_test_seq, y_test_seq = create_sequences(
        X_test_norm, y_test,
        time_steps=TIME_STEPS,
        verbose=True
    )
    
    # ========================================================================
    # STEP 5: Validate Sequences
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Validating sequence shapes...")
    print("-" * 80)
    
    print("Training sequences:")
    validate_sequence_shape(X_train_seq, y_train_seq, TIME_STEPS, 6, verbose=True)
    
    print("\nTest sequences:")
    validate_sequence_shape(X_test_seq, y_test_seq, TIME_STEPS, 6, verbose=True)
    
    # ========================================================================
    # STEP 6: Build Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Building CNN-LSTM model...")
    print("-" * 80)
    
    model_config = {
        'cnn_filters_1': HYPERPARAMETERS['CNN_FILTERS_1'],
        'cnn_filters_2': HYPERPARAMETERS['CNN_FILTERS_2'],
        'cnn_kernel_size': HYPERPARAMETERS['CNN_KERNEL_SIZE'],
        'lstm_units': HYPERPARAMETERS['LSTM_UNITS'],
        'dropout_rate': HYPERPARAMETERS['DROPOUT_RATE'],
        'dense_units': HYPERPARAMETERS['DENSE_UNITS'],
        'learning_rate': HYPERPARAMETERS['LEARNING_RATE']
    }
    
    input_shape = (TIME_STEPS, 6)  # (timesteps, features)
    num_classes = len(label_encoder.classes_)
    
    model = build_cnn_lstm_model(input_shape, num_classes, config=model_config)
    # Use simple metrics to avoid Keras compatibility issues
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HYPERPARAMETERS['LEARNING_RATE']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print_model_summary(model, save_to_file=None)  # Skip file save due to encoding issues
    
    # ========================================================================
    # STEP 7: Setup Callbacks
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Setting up training callbacks...")
    print("-" * 80)
    
    callbacks = get_callbacks(
        model_save_path=HYPERPARAMETERS['MODEL_SAVE_PATH'],
        patience=HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']
    )
    
    for cb in callbacks:
        print(f"  ✓ {cb.__class__.__name__}")
    
    # ========================================================================
    # STEP 8: Train Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: Training model...")
    print("-" * 80)
    print(f"Epochs: {HYPERPARAMETERS['EPOCHS']}")
    print(f"Batch size: {HYPERPARAMETERS['BATCH_SIZE']}")
    print(f"Validation split: {HYPERPARAMETERS['VALIDATION_SPLIT']}")
    print()
    
    history = model.fit(
        X_train_seq, y_train_seq,
        batch_size=HYPERPARAMETERS['BATCH_SIZE'],
        epochs=HYPERPARAMETERS['EPOCHS'],
        validation_split=HYPERPARAMETERS['VALIDATION_SPLIT'],
        callbacks=callbacks,
        verbose=1
    )
    
    # ========================================================================
    # STEP 9: Save Artifacts
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: Saving training artifacts...")
    print("-" * 80)
    
    # Save hyperparameters
    save_hyperparameters(HYPERPARAMETERS, HYPERPARAMETERS['HYPERPARAMS_SAVE_PATH'])
    
    # Save training history
    save_training_history(history, HYPERPARAMETERS['HISTORY_SAVE_PATH'])
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(scaler, label_encoder, '../models')
    
    # Create quick summary plot
    plot_quick_summary(history)
    
    # ========================================================================
    # STEP 10: Quick Evaluation on Test Set
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 10: Quick evaluation on test set...")
    print("-" * 80)
    
    test_loss, test_acc = model.evaluate(
        X_test_seq, y_test_seq,
        verbose=0
    )
    
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Generated files:")
    print(f"  ✓ {HYPERPARAMETERS['MODEL_SAVE_PATH']} - Trained model")
    print(f"  ✓ {HYPERPARAMETERS['HISTORY_SAVE_PATH']} - Training history")
    print(f"  ✓ {HYPERPARAMETERS['HYPERPARAMS_SAVE_PATH']} - Hyperparameters")
    print(f"  ✓ scaler.pkl - Fitted scaler")
    print(f"  ✓ label_encoder.pkl - Label encoder")
    print(f"  ✓ model_architecture.txt - Model summary")
    print(f"  ✓ training_summary.png - Training plots")
    print()
    print("Next step: Run evaluate_model.py for comprehensive evaluation")
    print("=" * 80)


if __name__ == "__main__":
    main()
