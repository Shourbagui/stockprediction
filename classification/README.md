# Deep Learning Classification Model

## Overview
Multiclass classification model using neural networks to predict **Buy**, **Hold**, or **Sell** signals for US equities.

- **Target Variable**: `prediction_20d` (Buy/Hold/Sell)
- **Model**: Deep Neural Network (4 hidden layers with batch normalization and dropout)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Evaluation Metric**: Accuracy, Macro F1

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run training
```bash
cd /Users/alyshourbagui/bigdata/classification
python train_model.py
```

This will:
- Load and preprocess data
- Handle missing values and outliers
- Standardize features using StandardScaler
- Split into train (80%) and test (20%) sets with stratification
- Train a 4-layer neural network with early stopping
- Evaluate on test set
- Save model, scaler, and results

## Outputs

- `model.h5` - Trained Keras model
- `scaler.pkl` - Feature standardization object
- `label_encoder.pkl` - Label encoding object
- `results.json` - Test accuracy and F1 score
- `training_history.png` - Training and validation accuracy/loss curves
- `confusion_matrix.png` - Confusion matrix heatmap

## Model Architecture

```
Input (61 features)
↓
Dense(256, relu) + BatchNorm + Dropout(0.3)
↓
Dense(128, relu) + BatchNorm + Dropout(0.3)
↓
Dense(64, relu) + BatchNorm + Dropout(0.2)
↓
Dense(32, relu) + BatchNorm + Dropout(0.2)
↓
Dense(3, softmax) → Output (Buy/Hold/Sell)
```

## Key Parameters

- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Learning Rate**: 0.001
- **Early Stopping Patience**: 5 epochs
- **Dropout**: 0.2-0.3
- **Validation Split**: 10%

## Next Steps

- Hyperparameter tuning (layer sizes, dropout rates, learning rate)
- Class weight adjustment for imbalanced data
- Ensemble methods (averaging multiple models)
- Prediction script for new data

## Files

- `train_model.py` - Main training script
- `requirements.txt` - Python dependencies
- `README.md` - This file
