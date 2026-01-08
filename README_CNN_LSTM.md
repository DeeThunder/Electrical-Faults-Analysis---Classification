# CNN-LSTM Electrical Fault Classification

## Overview

This project implements a **Hybrid CNN-LSTM Deep Learning Pipeline** for classifying electrical faults in transmission lines using time-series voltage and current measurements.

**Pipeline Architecture:**
```
Raw CSV Data → Preprocessing → Time Windows → CNN (Feature Extraction) → LSTM (Temporal Learning) → Dense (Classification) → Fault Prediction
```

---

## Dataset

- **Source:** `classData.csv`
- **Samples:** 7,861 measurements
- **Features:** 6 (Ia, Ib, Ic, Va, Vb, Vc)
  - `Ia`, `Ib`, `Ic`: Current measurements for phases A, B, C
  - `Va`, `Vb`, `Vc`: Voltage measurements for phases A, B, C
- **Target:** 4 binary columns (G, C, B, A) combined into 6 fault types
- **Classes:**
  - `0000`: No Fault
  - `1001`: LG Fault (Line-Ground)
  - `0011`: LL Fault (Line-Line)
  - `1011`: LLG Fault (Line-Line-Ground)
  - `0111`: LLL Fault (Three-phase)
  - `1111`: LLLG Fault (Three-phase symmetrical)

---

## Installation

### Requirements

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

**Recommended versions:**
- Python 3.8+
- TensorFlow 2.10+
- NumPy 1.23+
- Pandas 1.5+
- Scikit-learn 1.2+

### GPU Support (Optional but Recommended)

For faster training, install TensorFlow with GPU support:
```bash
pip install tensorflow-gpu
```

---

## Project Structure

```
samuel CNN/
├── classData.csv                    # Dataset
├── cnn_lstm_preprocessing.py        # Data preprocessing module
├── cnn_lstm_model.py                # Model architecture module
├── train_cnn_lstm.py                # Training script
├── evaluate_model.py                # Evaluation script
├── README_CNN_LSTM.md               # This file
│
├── best_cnn_lstm_model.h5           # Trained model (generated)
├── scaler.pkl                       # Fitted scaler (generated)
├── label_encoder.pkl                # Label encoder (generated)
├── training_history.csv             # Training metrics (generated)
├── hyperparameters_log.json         # Hyperparameters (generated)
│
└── Visualizations (generated):
    ├── confusion_matrix.png
    ├── training_history.png
    ├── classwise_f1_scores.png
    └── training_summary.png
```

---

## Usage

### 1. Test Preprocessing Module

```bash
cd "c:\Users\HP\Downloads\samuel CNN"
python cnn_lstm_preprocessing.py
```

**Expected output:**
- Data loading confirmation
- Sequence generation statistics
- Shape validation results

### 2. Test Model Architecture

```bash
python cnn_lstm_model.py
```

**Expected output:**
- Model architecture summary
- Parameter counts
- Quick training test results

### 3. Train the Model

```bash
python train_cnn_lstm.py
```

**Training process:**
1. Loads and preprocesses data
2. Creates time-series sequences (TIME_STEPS=10)
3. Builds CNN-LSTM model
4. Trains with early stopping and learning rate reduction
5. Saves best model and training artifacts

**Expected duration:**
- CPU: 5-15 minutes
- GPU: 1-3 minutes

**Generated files:**
- `best_cnn_lstm_model.h5`
- `training_history.csv`
- `hyperparameters_log.json`
- `scaler.pkl`
- `label_encoder.pkl`
- `model_architecture.txt`
- `training_summary.png`

### 4. Evaluate the Model

```bash
python evaluate_model.py
```

**Evaluation includes:**
- Overall accuracy, precision, recall, F1-score
- Class-wise performance metrics
- Confusion matrix
- Training history visualization
- Class-wise F1 score comparison

**Generated files:**
- `confusion_matrix.png`
- `training_history.png`
- `classwise_f1_scores.png`
- `evaluation_metrics.txt`

---

## Hyperparameters

### Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Data** |
| `TIME_STEPS` | 10 | Sequence length for sliding window |
| `TEST_SIZE` | 0.2 | Test set proportion |
| `VALIDATION_SPLIT` | 0.2 | Validation split during training |
| **Model Architecture** |
| `CNN_FILTERS_1` | 64 | First Conv1D layer filters |
| `CNN_FILTERS_2` | 128 | Second Conv1D layer filters |
| `CNN_KERNEL_SIZE` | 3 | Convolutional kernel size |
| `LSTM_UNITS` | 100 | LSTM layer units |
| `DROPOUT_RATE` | 0.3 | Dropout rate for regularization |
| `DENSE_UNITS` | 64 | Dense layer units |
| **Training** |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `EARLY_STOPPING_PATIENCE` | 10 | Early stopping patience |

### Tuning Hyperparameters

Edit the `HYPERPARAMETERS` dictionary in `train_cnn_lstm.py`:

```python
HYPERPARAMETERS = {
    'TIME_STEPS': 15,        # Try 5, 10, 15, 20
    'LSTM_UNITS': 150,       # Try 50, 100, 150, 200
    'BATCH_SIZE': 64,        # Try 16, 32, 64, 128
    # ... other parameters
}
```

---

## Model Architecture

### Standard CNN-LSTM Model

```
Input: (10 timesteps, 6 features)
    ↓
Conv1D (64 filters, kernel=3, ReLU)
    ↓
BatchNormalization
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.3)
    ↓
Conv1D (128 filters, kernel=3, ReLU)
    ↓
BatchNormalization
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.3)
    ↓
LSTM (100 units, dropout=0.3, recurrent_dropout=0.2)
    ↓
Dense (64 units, ReLU)
    ↓
Dropout (0.4)
    ↓
Dense (6 units, Softmax)
    ↓
Output: Fault class probabilities
```

**Total Parameters:** ~150,000 (varies with configuration)

### Alternative Architectures

The `cnn_lstm_model.py` module also provides:
- **Lightweight model:** Faster training, fewer parameters
- **Deep model:** More layers, potentially higher accuracy

---

## Expected Results

### Performance Targets

- **Accuracy:** ≥95%
- **Precision:** ≥94%
- **Recall:** ≥94%
- **F1-Score:** ≥94%

### Baseline Comparison

| Model | Accuracy | Training Time | Parameters |
|-------|----------|---------------|------------|
| RandomForest (baseline) | ~99% | <1 min | N/A |
| CNN-LSTM (this project) | ~95-98% | 5-15 min | ~150K |

**Note:** The CNN-LSTM model demonstrates deep learning techniques for time-series classification. While RandomForest may achieve higher accuracy on this dataset, the CNN-LSTM approach is valuable for:
- Learning temporal patterns
- Handling sequential data
- Demonstrating modern deep learning architectures

---

## Troubleshooting

### Memory Errors

**Problem:** Out of memory during training

**Solutions:**
- Reduce `BATCH_SIZE` (try 16 or 8)
- Reduce `TIME_STEPS` (try 5 or 7)
- Use lightweight model variant

### Poor Accuracy

**Problem:** Model accuracy < 90%

**Solutions:**
- Increase `EPOCHS` (try 100)
- Adjust `LEARNING_RATE` (try 0.0005 or 0.002)
- Increase model complexity (use deep model variant)
- Increase `TIME_STEPS` (try 15 or 20)

### Overfitting

**Problem:** Training accuracy >> Validation accuracy

**Solutions:**
- Increase `DROPOUT_RATE` (try 0.4 or 0.5)
- Add L2 regularization
- Reduce model complexity
- Increase training data (if possible)

### Slow Training

**Problem:** Training takes too long

**Solutions:**
- Use GPU acceleration
- Reduce `BATCH_SIZE`
- Use lightweight model variant
- Reduce `EPOCHS` with early stopping

---

## Reproducibility

All experiments are reproducible using:
1. **Random seeds:** Set in `train_cnn_lstm.py` (NumPy and TensorFlow)
2. **Hyperparameter logging:** Saved in `hyperparameters_log.json`
3. **Preprocessing artifacts:** `scaler.pkl` and `label_encoder.pkl`
4. **Model checkpoints:** Best model saved during training

To reproduce results:
```bash
python train_cnn_lstm.py  # Uses fixed random_state=42
python evaluate_model.py
```

---

## Integration with Original Notebook

To add CNN-LSTM results to `notebook6c7b88094f.py`:

```python
# Add after the RandomForest multiclass classification section

# ============================================================================
# CNN-LSTM HYBRID MODEL
# ============================================================================

from cnn_lstm_preprocessing import load_and_prepare_data, normalize_features, create_sequences
from cnn_lstm_model import build_cnn_lstm_model, compile_model
import tensorflow as tf

# Load and preprocess
X, y, le = load_and_prepare_data('classData.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)

# Create sequences
TIME_STEPS = 10
X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test, TIME_STEPS)

# Build and train
model = build_cnn_lstm_model((TIME_STEPS, 6), 6)
model = compile_model(model)
history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate
y_pred = np.argmax(model.predict(X_test_seq), axis=1)
print(f"CNN-LSTM Accuracy: {accuracy_score(y_test_seq, y_pred):.4f}")
```

---

## Term Paper Integration

### Key Points to Include

1. **Problem Statement:** Electrical fault classification using time-series data
2. **Methodology:** Hybrid CNN-LSTM architecture for temporal pattern recognition
3. **Pipeline:** Data preprocessing → Sequence generation → CNN feature extraction → LSTM temporal learning → Classification
4. **Results:** Accuracy, precision, recall, F1-score, confusion matrix
5. **Comparison:** CNN-LSTM vs. RandomForest baseline
6. **Conclusion:** Deep learning approach demonstrates effective temporal pattern learning

### Figures to Include

- Pipeline architecture diagram
- Model architecture summary
- Confusion matrix
- Training history curves
- Class-wise F1 score comparison

---

## References

- **TensorFlow Documentation:** https://www.tensorflow.org/
- **Keras Sequential API:** https://keras.io/guides/sequential_model/
- **Time Series Classification:** https://arxiv.org/abs/1809.04356

---

## License

This project is for educational purposes as part of a term paper on electrical fault classification.

---

## Contact

For questions or issues, please refer to the implementation plan or consult the course instructor.
