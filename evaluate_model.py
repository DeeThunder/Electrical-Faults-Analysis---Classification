"""
CNN-LSTM Model Evaluation Script
=================================
Comprehensive evaluation and visualization for trained CNN-LSTM model.

Usage:
    python evaluate_model.py
    
This script will:
    1. Load trained model and preprocessing artifacts
    2. Generate predictions on test set
    3. Compute comprehensive metrics
    4. Create visualizations (confusion matrix, training curves, class-wise analysis)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras

# Import custom modules
from cnn_lstm_preprocessing import (
    load_and_prepare_data,
    normalize_features,
    create_sequences,
    load_preprocessing_artifacts
)
from sklearn.model_selection import train_test_split

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_trained_model(model_path='best_cnn_lstm_model.h5'):
    """Load the trained CNN-LSTM model."""
    model = keras.models.load_model(model_path)
    print(f"✓ Loaded model from {model_path}")
    return model


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Create and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - CNN-LSTM Model', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(history_path='training_history.csv', save_path='training_history.png'):
    """Plot training and validation metrics over epochs."""
    history_df = pd.read_csv(history_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history_df['epoch'], history_df['accuracy'], 
                    label='Train', linewidth=2, marker='o', markersize=4)
    ax1.plot(history_df['epoch'], history_df['val_accuracy'], 
                    label='Validation', linewidth=2, marker='s', markersize=4)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history_df['epoch'], history_df['loss'], 
                    label='Train', linewidth=2, marker='o', markersize=4)
    ax2.plot(history_df['epoch'], history_df['val_loss'], 
                    label='Validation', linewidth=2, marker='s', markersize=4)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()


def plot_classwise_f1_scores(y_true, y_pred, class_names, save_path='classwise_f1_scores.png'):
    """Create bar plot of class-wise F1 scores."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract F1 scores
    f1_scores = {cls: metrics['f1-score'] 
                 for cls, metrics in report.items() 
                 if cls not in ['accuracy', 'macro avg', 'weighted avg']}
    
    # Create DataFrame for plotting
    f1_df = pd.DataFrame({
        'Fault Type': list(f1_scores.keys()),
        'F1 Score': list(f1_scores.values())
    })
    f1_df = f1_df.sort_values('F1 Score', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(f1_df)))
    bars = plt.barh(f1_df['Fault Type'], f1_df['F1 Score'], color=colors)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, f1_df['F1 Score'])):
        plt.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')
    
    plt.xlabel('F1 Score', fontsize=12)
    plt.ylabel('Fault Type', fontsize=12)
    plt.title('Class-wise F1 Scores - CNN-LSTM Model', fontsize=14, fontweight='bold', pad=20)
    plt.xlim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class-wise F1 scores plot saved to {save_path}")
    plt.close()


def print_detailed_metrics(y_true, y_pred, class_names):
    """Print comprehensive classification metrics."""
    print("=" * 80)
    print("DETAILED CLASSIFICATION METRICS")
    print("=" * 80)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\nOVERALL METRICS:")
    print("-" * 80)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Class-wise report
    print("\n" + "=" * 80)
    print("CLASS-WISE CLASSIFICATION REPORT:")
    print("-" * 80)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion matrix statistics
    cm = confusion_matrix(y_true, y_pred)
    print("=" * 80)
    print("CONFUSION MATRIX STATISTICS:")
    print("-" * 80)
    for i, class_name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i, i]
        class_acc = correct / total if total > 0 else 0
        print(f"{class_name:8s}: {correct:4d}/{total:4d} correct ({class_acc*100:6.2f}%)")


def save_metrics_to_file(y_true, y_pred, class_names, filepath='evaluation_metrics.txt'):
    """Save all metrics to a text file."""
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CNN-LSTM MODEL EVALUATION METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n\n")
        
        # Classification report
        f.write("=" * 80 + "\n")
        f.write("CLASS-WISE REPORT:\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    print(f"✓ Metrics saved to {filepath}")


def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("CNN-LSTM MODEL EVALUATION")
    print("=" * 80)
    print()
    
    # ========================================================================
    # STEP 1: Load Data and Preprocessing Artifacts
    # ========================================================================
    print("STEP 1: Loading data and preprocessing artifacts...")
    print("-" * 80)
    
    # Load data
    X, y, label_encoder = load_and_prepare_data('classData.csv', verbose=False)
    class_names = label_encoder.classes_
    
    # Split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Load preprocessing artifacts
    scaler, _ = load_preprocessing_artifacts()
    
    # Normalize
    _, X_test_norm, _ = normalize_features(X_train, X_test, method='standard', verbose=False)
    
    # Create sequences
    TIME_STEPS = 10
    X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test, TIME_STEPS, verbose=False)
    
    print(f"✓ Test sequences shape: {X_test_seq.shape}")
    print(f"✓ Number of classes: {len(class_names)}")
    
    # ========================================================================
    # STEP 2: Load Trained Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Loading trained model...")
    print("-" * 80)
    
    model = load_trained_model('best_cnn_lstm_model.h5')
    
    # ========================================================================
    # STEP 3: Generate Predictions
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Generating predictions...")
    print("-" * 80)
    
    y_pred_probs = model.predict(X_test_seq, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print(f"✓ Generated predictions for {len(y_pred)} samples")
    
    # ========================================================================
    # STEP 4: Compute Metrics
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Computing metrics...")
    print("-" * 80)
    
    print_detailed_metrics(y_test_seq, y_pred, class_names)
    
    # ========================================================================
    # STEP 5: Create Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Creating visualizations...")
    print("-" * 80)
    
    plot_confusion_matrix(y_test_seq, y_pred, class_names)
    plot_training_history()
    plot_classwise_f1_scores(y_test_seq, y_pred, class_names)
    
    # ========================================================================
    # STEP 6: Save Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Saving results...")
    print("-" * 80)
    
    save_metrics_to_file(y_test_seq, y_pred, class_names)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  ✓ confusion_matrix.png - Confusion matrix heatmap")
    print("  ✓ training_history.png - Training curves")
    print("  ✓ classwise_f1_scores.png - Class-wise F1 scores")
    print("  ✓ evaluation_metrics.txt - Detailed metrics report")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
