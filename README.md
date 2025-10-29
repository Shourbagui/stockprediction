# Stock Market Classification Project

## Overview
A machine learning pipeline for stock market classification (Buy/Hold/Sell) using historical price data, technical indicators, and sentiment features.

## Quick Start

After cloning the repository, follow these steps:

### 1. Clone and navigate
```bash
git clone https://github.com/Shourbagui/stockprediction.git
cd stockprediction
```

### 2. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Or install individually if preferred:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels ydata-profiling pyarrow fastparquet
```

### 4. Prepare your data
Place your `panel_daily.csv` file in the root directory. The CSV should contain:
- **Required columns**: `date`, `ticker`
- **Price data**: Open, High, Low, Close, Volume
- **Technical indicators**: Moving averages, RSI, momentum indicators, etc.
- **Labels**: `label_5d`, `label_20d` (values: "Buy", "Hold", "Sell")

### 5. Run the EDA pipeline
```bash
python eda/eda.py
```

This will generate:
- `reports/eda/index.html` - Modern interactive dashboard
- `reports/eda/ydata_profile.html` - Comprehensive data profile
- CSV summaries and PNG visualizations in `reports/eda/`

### 6. View the Dashboard
```bash
# From the project root directory
python3 -m http.server 8000
```

Then open your browser and visit:
- **Main Dashboard**: http://localhost:8000/reports/eda/index.html
- **Full Profile**: http://localhost:8000/reports/eda/ydata_profile.html

**Note**: You can use any available port (8001, 8002, etc.) if 8000 is already in use.

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

