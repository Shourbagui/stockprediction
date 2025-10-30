import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = Path("/Users/alyshourbagui/bigdata/panel_daily_with_msq.csv")
OUTPUT_DIR = Path("/Users/alyshourbagui/bigdata/classification")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)
df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique tickers: {df['ticker'].nunique()}")

# ============================================================================
# TARGET VARIABLE
# ============================================================================
TARGET = "prediction_20d"
print(f"\nTarget variable: {TARGET}")
print(f"Class distribution:\n{df[TARGET].value_counts()}\n")
print(f"Class distribution (%):\n{df[TARGET].value_counts(normalize=True) * 100}\n")

# Drop rows with missing target
df = df.dropna(subset=[TARGET])
print(f"Data shape after dropping NaN targets: {df.shape}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by ticker and date for proper time-series feature creation
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

# 1. Temporal features
print("\n1. Creating temporal features...")
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['day_of_year'] = df['date'].dt.dayofyear

# 2. Lagged features (per ticker)
print("2. Creating lagged features...")
lag_periods = [1, 5, 10, 20]
lag_features = ['Close', 'Volume', 'ret_1d', 'rsi14']

for col in lag_features:
    if col in df.columns:
        for lag in lag_periods:
            df[f'{col}_lag_{lag}'] = df.groupby('ticker')[col].shift(lag)

# 3. Rolling statistics (per ticker)
print("3. Creating rolling statistics...")
rolling_windows = [5, 10, 20]
rolling_features = ['Close', 'Volume', 'ret_1d']

for col in rolling_features:
    if col in df.columns:
        for window in rolling_windows:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}'] = df.groupby('ticker')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            # Rolling std (volatility)
            df[f'{col}_rolling_std_{window}'] = df.groupby('ticker')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

# 4. Price momentum features
print("4. Creating momentum features...")
df['price_momentum_5'] = (df['Close'] - df['Close'].groupby(df['ticker']).shift(5)) / df['Close'].groupby(df['ticker']).shift(5)
df['price_momentum_20'] = (df['Close'] - df['Close'].groupby(df['ticker']).shift(20)) / df['Close'].groupby(df['ticker']).shift(20)

# 5. Volume features
print("5. Creating volume features...")
df['volume_ratio_5'] = df['Volume'] / df.groupby('ticker')['Volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['volume_ratio_20'] = df['Volume'] / df.groupby('ticker')['Volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())

# 6. Encode ticker as categorical
print("6. Encoding ticker...")
ticker_encoder = LabelEncoder()
df['ticker_encoded'] = ticker_encoder.fit_transform(df['ticker'])

# 7. Volatility regime
print("7. Creating volatility regime...")
df['volatility_regime'] = pd.cut(df['vol_z20'], bins=[-np.inf, -1, 1, np.inf], labels=[0, 1, 2])

print(f"\nTotal features after engineering: {df.shape[1]}")

# ============================================================================
# PREPARE FEATURES AND TARGET
# ============================================================================
print("\n" + "="*70)
print("PREPARING FEATURES")
print("="*70)

# Exclude non-feature columns
exclude_cols = [
    "date", "ticker", TARGET, "label_5d", "fwd_ret_5d", "fwd_ret_20d"
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"Number of features: {len(feature_cols)}")

# Separate features and target
X = df[feature_cols].copy()
y = df[TARGET].copy()

# Replace inf with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Fill NaN with median (per column) - handle categorical and numeric separately
print("\nHandling missing values...")
numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(exclude=[np.number]).columns

# Fill numeric columns with median
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

# Fill categorical columns with mode or convert to numeric
for col in categorical_cols:
    if X[col].dtype.name == 'category':
        # Convert category to numeric codes
        X[col] = X[col].cat.codes
        # Fill any remaining NaNs with mode
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)

print(f"Missing values after imputation: {X.isna().sum().sum()}")

# ============================================================================
# ENCODE TARGET
# ============================================================================
print("\n" + "="*70)
print("ENCODING TARGET")
print("="*70)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")
print(f"Encoded class distribution:\n{pd.Series(y_encoded).value_counts().sort_index()}")

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print(f"\nClass weights: {class_weights_dict}")

# ============================================================================
# TIME-AWARE TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("TRAIN-TEST SPLIT (Time-aware)")
print("="*70)

# Time-based split (use last 20% as test set)
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y_encoded[:split_idx]
y_test = y_encoded[split_idx:]

print(f"Train set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nTrain class distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"\nTest class distribution:\n{pd.Series(y_test).value_counts().sort_index()}")

# ============================================================================
# FEATURE SCALING
# ============================================================================
print("\n" + "="*70)
print("FEATURE SCALING")
print("="*70)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

# ============================================================================
# BUILD GRADIENT BOOSTING MODEL
# ============================================================================
print("\n" + "="*70)
print("TRAINING GRADIENT BOOSTING MODEL")
print("="*70)

# Calculate sample weights for class imbalance
sample_weights = np.array([class_weights_dict[label] for label in y_train])

# Create model with parameters optimized for class imbalance
# HistGradientBoostingClassifier handles NaN natively and is much faster
model = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,
    max_depth=8,
    min_samples_leaf=100,
    l2_regularization=0.1,
    max_bins=255,
    random_state=RANDOM_STATE,
    verbose=1,
    validation_fraction=VALIDATION_SIZE,
    n_iter_no_change=50,
    tol=1e-4,
    class_weight=class_weights_dict
)

print("Model parameters:")
print(f"  max_iter: {model.max_iter}")
print(f"  learning_rate: {model.learning_rate}")
print(f"  max_depth: {model.max_depth}")
print(f"  min_samples_leaf: {model.min_samples_leaf}")
print(f"  l2_regularization: {model.l2_regularization}")
print(f"  max_bins: {model.max_bins}")

# Train model (class weights already set in model)
print("\nTraining...")
model.fit(X_train_scaled, y_train)

print(f"\nTotal iterations trained: {model.n_iter_}")
print(f"Training complete!")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

# Predict
y_pred_probs = model.predict_proba(X_test_scaled)
y_pred = model.predict(X_test_scaled)

# Metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_f1_macro = f1_score(y_test, y_pred, average='macro')
test_f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_f1_macro:.4f}")
print(f"Test Weighted F1: {test_f1_weighted:.4f}")

print("\nClassification Report:")
report_str = classification_report(y_test, y_pred, target_names=class_names)
print(report_str)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ============================================================================
# FEATURE IMPORTANCE (Using permutation importance)
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

from sklearn.inspection import permutation_importance

# Calculate permutation importance on a subset for speed
print("Calculating permutation importance...")
perm_importance = permutation_importance(
    model, X_test_scaled[:1000], y_test[:1000],
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=1  # Use single core to avoid parallel processing issues
)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importance.head(20).to_string(index=False))

# Save feature importance
feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
print(f"\nFeature importance saved to {OUTPUT_DIR / 'feature_importance.csv'}")

# ============================================================================
# SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n" + "="*70)
print("SAVING MODEL AND ARTIFACTS")
print("="*70)

# Save Gradient Boosting model
with open(OUTPUT_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {OUTPUT_DIR / 'model.pkl'}")

# Save scaler
with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {OUTPUT_DIR / 'scaler.pkl'}")

# Save label encoder
with open(OUTPUT_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"Label encoder saved to {OUTPUT_DIR / 'label_encoder.pkl'}")

# Save ticker encoder
with open(OUTPUT_DIR / "ticker_encoder.pkl", "wb") as f:
    pickle.dump(ticker_encoder, f)
print(f"Ticker encoder saved to {OUTPUT_DIR / 'ticker_encoder.pkl'}")

# Save feature columns
with open(OUTPUT_DIR / "feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print(f"Feature columns saved to {OUTPUT_DIR / 'feature_cols.pkl'}")

# Save results
results = {
    "test_accuracy": float(test_accuracy),
    "test_f1_macro": float(test_f1_macro),
    "test_f1_weighted": float(test_f1_weighted),
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "num_features": int(X_train.shape[1]),
    "num_classes": int(num_classes),
    "classes": list(class_names),
    "n_iterations": int(model.n_iter_),
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR / 'results.json'}")

# Save classification report as JSON
report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
with open(OUTPUT_DIR / "classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=2)
print(f"Classification report saved to {OUTPUT_DIR / 'classification_report.json'}")

# ============================================================================
# PLOT FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Feature importance plot
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance (Gain)')
plt.title('Top 20 Feature Importance (Gradient Boosting)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Feature importance plot saved to {OUTPUT_DIR / 'feature_importance.png'}")

# ============================================================================
# PLOT CONFUSION MATRIX
# ============================================================================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, 
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Test Accuracy: {test_accuracy:.4f})')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Confusion matrix plot saved to {OUTPUT_DIR / 'confusion_matrix.png'}")

# ============================================================================
# PLOT CLASS DISTRIBUTION COMPARISON
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# True distribution
true_counts = pd.Series(y_test).value_counts().sort_index()
axes[0].bar(range(len(true_counts)), true_counts.values, color='steelblue')
axes[0].set_xticks(range(len(class_names)))
axes[0].set_xticklabels(class_names)
axes[0].set_ylabel('Count')
axes[0].set_title('True Class Distribution (Test Set)')
axes[0].grid(alpha=0.3, axis='y')

# Predicted distribution
pred_counts = pd.Series(y_pred).value_counts().sort_index()
axes[1].bar(range(len(pred_counts)), pred_counts.values, color='coral')
axes[1].set_xticks(range(len(class_names)))
axes[1].set_xticklabels(class_names)
axes[1].set_ylabel('Count')
axes[1].set_title('Predicted Class Distribution (Test Set)')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Class distribution comparison saved to {OUTPUT_DIR / 'class_distribution_comparison.png'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("GRADIENT BOOSTING CLASSIFICATION MODEL TRAINING COMPLETE")
print("="*70)
print(f"Model Type: Scikit-learn Gradient Boosting Classifier")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_f1_macro:.4f}")
print(f"Test Weighted F1: {test_f1_weighted:.4f}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Total iterations: {model.n_iter_}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*70)
