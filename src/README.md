# Source Code Directory

This directory contains all Python modules for the CNN-LSTM electrical fault classification pipeline.

## Modules

### 1. `cnn_lstm_preprocessing.py`
**Purpose:** Data preprocessing and sequence generation

**Key Functions:**
- `load_and_prepare_data()` - Load CSV and encode labels
- `normalize_features()` - StandardScaler normalization
- `create_sequences()` - Sliding window sequence generation
- `validate_sequence_shape()` - Data integrity validation
- `save_preprocessing_artifacts()` - Save scaler and encoder

**Usage:**
```python
from cnn_lstm_preprocessing import load_and_prepare_data, normalize_features, create_sequences

# Load data
X, y, label_encoder = load_and_prepare_data('../data/classData.csv')

# Normalize
X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)

# Create sequences
X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train, time_steps=10)
```

---

### 2. `cnn_lstm_model.py`
**Purpose:** Model architecture definitions

**Key Functions:**
- `build_cnn_lstm_model()` - Build standard CNN-LSTM model
- `build_lightweight_model()` - Build lighter variant
- `build_deep_model()` - Build deeper variant
- `compile_model()` - Compile with optimizer and metrics
- `get_callbacks()` - Create training callbacks
- `print_model_summary()` - Display architecture

**Usage:**
```python
from cnn_lstm_model import build_cnn_lstm_model, compile_model, get_callbacks

# Build model
model = build_cnn_lstm_model(input_shape=(10, 6), num_classes=6)

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Get callbacks
callbacks = get_callbacks(model_save_path='../models/best_model.h5')
```

---

### 3. `train_cnn_lstm.py`
**Purpose:** Complete training pipeline

**Execution:**
```bash
cd src
python train_cnn_lstm.py
```

**Process:**
1. Loads data from `../data/classData.csv`
2. Preprocesses and creates sequences
3. Builds and compiles model
4. Trains with callbacks
5. Saves artifacts to `../models/` and `../results/`

**Outputs:**
- `../models/best_cnn_lstm_model.h5` - Trained model
- `../models/scaler.pkl` - Fitted scaler
- `../models/label_encoder.pkl` - Label encoder
- `../results/metrics/training_history.csv` - Training metrics
- `../results/metrics/hyperparameters_log.json` - Hyperparameters
- `../results/visualizations/training_summary.png` - Quick plot

---

### 4. `evaluate_model.py`
**Purpose:** Model evaluation and visualization

**Execution:**
```bash
cd src
python evaluate_model.py
```

**Process:**
1. Loads trained model from `../models/`
2. Generates predictions on test set
3. Computes comprehensive metrics
4. Creates visualizations

**Outputs:**
- `../results/visualizations/confusion_matrix.png`
- `../results/visualizations/training_history.png`
- `../results/visualizations/classwise_f1_scores.png`
- `../results/metrics/evaluation_metrics.txt`

---

## Running the Pipeline

### Step 1: Train the Model
```bash
cd src
$env:PYTHONIOENCODING='utf-8'  # Windows only
python train_cnn_lstm.py
```

### Step 2: Evaluate the Model
```bash
python evaluate_model.py
```

### Step 3: View Results
- Check `../results/visualizations/` for plots
- Check `../results/metrics/` for detailed metrics
- Check `../models/` for trained artifacts

---

## Module Dependencies

```
cnn_lstm_preprocessing.py
    ↓
train_cnn_lstm.py → cnn_lstm_model.py
    ↓
evaluate_model.py
```

All modules are standalone and can be imported independently.

---

## Configuration

Edit hyperparameters in `train_cnn_lstm.py`:

```python
HYPERPARAMETERS = {
    'TIME_STEPS': 10,           # Sequence length
    'BATCH_SIZE': 32,           # Training batch size
    'EPOCHS': 50,               # Maximum epochs
    'LEARNING_RATE': 0.001,     # Adam learning rate
    'CNN_FILTERS_1': 64,        # First Conv1D filters
    'CNN_FILTERS_2': 128,       # Second Conv1D filters
    'LSTM_UNITS': 100,          # LSTM units
    'DROPOUT_RATE': 0.3,        # Dropout rate
}
```

---

## Testing Individual Modules

Each module has a `if __name__ == "__main__":` block for testing:

```bash
# Test preprocessing
python cnn_lstm_preprocessing.py

# Test model architecture
python cnn_lstm_model.py
```
