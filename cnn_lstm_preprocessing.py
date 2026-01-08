"""
CNN-LSTM Data Preprocessing Module
====================================
Handles data loading, preprocessing, and time-series sequence generation
for electrical fault classification.

Pipeline: Raw CSV → Feature/Label Separation → Normalization → Sequence Generation → 3D Arrays
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


def load_and_prepare_data(filepath, verbose=True):
    """
    Load electrical fault dataset and prepare features and labels.
    
    Args:
        filepath (str): Path to CSV file (e.g., 'classData.csv')
        verbose (bool): Print diagnostic information
    
    Returns:
        tuple: (X, y, label_encoder)
            - X: numpy array of shape (n_samples, n_features)
            - y: numpy array of encoded labels (n_samples,)
            - label_encoder: fitted LabelEncoder object
    """
    # Load data
    df = pd.read_csv(filepath)
    
    if verbose:
        print(f"✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Check for missing values
    if df.isna().sum().sum() > 0:
        print(f"⚠ Warning: Found {df.isna().sum().sum()} missing values")
        df = df.dropna()
        print(f"  Dropped rows with missing values. New shape: {df.shape}")
    
    # Separate features and target columns
    feature_cols = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
    target_cols = ['G', 'C', 'B', 'A']
    
    X = df[feature_cols].values
    
    # Create combined fault type label (e.g., "1001" for LG fault)
    df['faultType'] = (df['G'].astype(str) + 
                       df['C'].astype(str) + 
                       df['B'].astype(str) + 
                       df['A'].astype(str))
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['faultType'])
    
    if verbose:
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Labels shape: {y.shape}")
        print(f"✓ Number of classes: {len(le.classes_)}")
        print(f"  Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
        print(f"  Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"    {le.inverse_transform([cls])[0]}: {count} samples")
    
    return X, y, le


def normalize_features(X_train, X_test, method='standard', verbose=True):
    """
    Normalize feature values using StandardScaler or MinMaxScaler.
    
    Args:
        X_train: Training features (n_samples, n_features)
        X_test: Test features (n_samples, n_features)
        method: 'standard' for StandardScaler or 'minmax' for MinMaxScaler
        verbose: Print diagnostic information
    
    Returns:
        tuple: (X_train_normalized, X_test_normalized, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    
    # Fit on training data only
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    if verbose:
        print(f"✓ Normalization method: {method}")
        print(f"  Train - Mean: {X_train_norm.mean():.4f}, Std: {X_train_norm.std():.4f}")
        print(f"  Test  - Mean: {X_test_norm.mean():.4f}, Std: {X_test_norm.std():.4f}")
    
    return X_train_norm, X_test_norm, scaler


def create_sequences(X, y, time_steps=10, verbose=True):
    """
    Create time-series sequences using sliding window approach.
    
    Converts 2D feature array into 3D sequences for CNN-LSTM input.
    Each sequence contains 'time_steps' consecutive measurements.
    
    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        time_steps: Number of time steps in each sequence
        verbose: Print diagnostic information
    
    Returns:
        tuple: (X_sequences, y_sequences)
            - X_sequences: shape (n_sequences, time_steps, n_features)
            - y_sequences: shape (n_sequences,) - label from last timestep
    
    Example:
        If X has 100 samples and time_steps=10:
        - Sequence 0: samples [0:10], label from sample 9
        - Sequence 1: samples [1:11], label from sample 10
        - ...
        - Total sequences: 100 - 10 + 1 = 91
    """
    X_seq = []
    y_seq = []
    
    # Sliding window
    for i in range(len(X) - time_steps + 1):
        # Extract sequence of 'time_steps' consecutive samples
        X_seq.append(X[i:i + time_steps])
        # Use label from the last time step
        y_seq.append(y[i + time_steps - 1])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    if verbose:
        print(f"✓ Created sequences with time_steps={time_steps}")
        print(f"  Input shape: {X.shape} → Output shape: {X_seq.shape}")
        print(f"  Labels shape: {y.shape} → Output shape: {y_seq.shape}")
        print(f"  Samples reduced: {len(X)} → {len(X_seq)} ({len(X_seq)/len(X)*100:.1f}%)")
    
    return X_seq, y_seq


def validate_sequence_shape(X_seq, y_seq, expected_timesteps, expected_features, verbose=True):
    """
    Validate that sequences have correct shape and no invalid values.
    
    Args:
        X_seq: Sequence array (n_sequences, timesteps, features)
        y_seq: Label array (n_sequences,)
        expected_timesteps: Expected number of timesteps
        expected_features: Expected number of features
        verbose: Print diagnostic information
    
    Raises:
        AssertionError: If validation fails
    """
    # Check dimensions
    assert X_seq.ndim == 3, f"X_seq must be 3D, got {X_seq.ndim}D"
    assert y_seq.ndim == 1, f"y_seq must be 1D, got {y_seq.ndim}D"
    
    # Check shape
    n_sequences, timesteps, features = X_seq.shape
    assert timesteps == expected_timesteps, f"Expected {expected_timesteps} timesteps, got {timesteps}"
    assert features == expected_features, f"Expected {expected_features} features, got {features}"
    assert len(y_seq) == n_sequences, f"Mismatch: {n_sequences} sequences but {len(y_seq)} labels"
    
    # Check for invalid values
    assert not np.isnan(X_seq).any(), "X_seq contains NaN values"
    assert not np.isinf(X_seq).any(), "X_seq contains infinite values"
    assert not np.isnan(y_seq).any(), "y_seq contains NaN values"
    
    # Check label distribution
    unique_labels, counts = np.unique(y_seq, return_counts=True)
    
    if verbose:
        print(f"✓ Sequence validation passed")
        print(f"  Shape: {X_seq.shape}")
        print(f"  Value range: [{X_seq.min():.4f}, {X_seq.max():.4f}]")
        print(f"  Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"    Class {label}: {count} sequences ({count/len(y_seq)*100:.1f}%)")


def save_preprocessing_artifacts(scaler, label_encoder, output_dir='.'):
    """
    Save fitted scaler and label encoder for deployment.
    
    Args:
        scaler: Fitted StandardScaler or MinMaxScaler
        label_encoder: Fitted LabelEncoder
        output_dir: Directory to save artifacts
    """
    import os
    
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"✓ Saved preprocessing artifacts:")
    print(f"  - {scaler_path}")
    print(f"  - {encoder_path}")


def load_preprocessing_artifacts(output_dir='.'):
    """
    Load saved scaler and label encoder.
    
    Args:
        output_dir: Directory containing artifacts
    
    Returns:
        tuple: (scaler, label_encoder)
    """
    import os
    
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"✓ Loaded preprocessing artifacts from {output_dir}")
    
    return scaler, label_encoder


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("CNN-LSTM Preprocessing Module - Test Run")
    print("=" * 60)
    
    # Test with actual data
    try:
        # Load data
        X, y, le = load_and_prepare_data('classData.csv')
        
        # Split data
        print("\n" + "=" * 60)
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Normalize
        print("\n" + "=" * 60)
        print("Normalizing features...")
        X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)
        
        # Create sequences
        print("\n" + "=" * 60)
        print("Creating sequences...")
        TIME_STEPS = 10
        X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train, TIME_STEPS)
        X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test, TIME_STEPS)
        
        # Validate
        print("\n" + "=" * 60)
        print("Validating sequences...")
        validate_sequence_shape(X_train_seq, y_train_seq, TIME_STEPS, 6)
        validate_sequence_shape(X_test_seq, y_test_seq, TIME_STEPS, 6)
        
        # Save artifacts
        print("\n" + "=" * 60)
        print("Saving preprocessing artifacts...")
        save_preprocessing_artifacts(scaler, le)
        
        print("\n" + "=" * 60)
        print("✓ All preprocessing tests passed!")
        print("=" * 60)
        
    except FileNotFoundError:
        print("\n⚠ classData.csv not found in current directory")
        print("  Run this script from the directory containing the dataset")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
