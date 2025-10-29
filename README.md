# Stock Market Classification Project

## Overview
A machine learning pipeline for stock market classification (Buy/Hold/Sell) using historical price data, technical indicators, and sentiment features.

## Setup

### 1. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels ydata-profiling pyarrow fastparquet
```

### 3. Data
- Place your `panel_daily.csv` in the root directory
- Expected columns: `date`, `ticker`, price data (Open, High, Low, Close), technical indicators, fundamentals, sentiment features
- Labels: `label_5d`, `label_20d` (Buy/Hold/Sell)

### 4. Run EDA
```bash
python eda/eda.py
```

This generates:
- `reports/eda/index.html` - Modern interactive dashboard
- `reports/eda/ydata_profile.html` - Full data profile
- CSV summaries of distributions, correlations, feature importance

### 5. View Dashboard
```bash
cd /Users/alyshourbagui/bigdata
python3 -m http.server 8000
# Visit: http://localhost:8000/reports/eda/index.html
```

## Project Structure
```
.
├── eda/
│   └── eda.py              # EDA pipeline script
├── reports/
│   └── eda/                # Generated EDA outputs
│       ├── index.html      # Main dashboard
│       └── *.png           # Visualization images
├── panel_daily.csv         # Raw data (not in repo)
└── .gitignore              # Excludes PNG, CSV, cache files
```

## Next Steps
- Implement data cleaning pipeline
- Feature engineering (time-series features, interaction terms)
- Train classification models (XGBoost, LightGBM, ensemble)
- Time-series cross-validation
- Model evaluation and deployment

