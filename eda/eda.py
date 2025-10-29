import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

# Configure Matplotlib for headless environments before importing pyplot
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

OUTPUT_DIR = Path("/Users/alyshourbagui/bigdata/reports/eda")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))
mpl.use("Agg", force=True)


DATA_PATH = Path("/Users/alyshourbagui/bigdata/panel_daily.csv")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(data_path: Path) -> pd.DataFrame:
    dtypes = {
        "ticker": "category",
    }
    df = pd.read_csv(
        data_path,
        dtype=dtypes,
        parse_dates=["date"],
        low_memory=False,
    )
    # Sort for time-aware operations
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def standardize_formats(df: pd.DataFrame) -> pd.DataFrame:
    # Ticker casing
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().astype("category")
    # Ensure date is datetime
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["date", "ticker"]).reset_index(drop=True)
    after = len(df)
    (OUTPUT_DIR / "deduplication.txt").write_text(f"Removed duplicates: {before - after}\n")
    return df


def summarize_missingness(df: pd.DataFrame) -> None:
    miss = df.isna().mean().sort_values(ascending=False)
    miss.to_csv(OUTPUT_DIR / "missingness_by_column.csv")

    # Heatmap (sample to keep plot legible)
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    plt.figure(figsize=(12, 6))
    sns.heatmap(sample.isna(), cbar=False)
    plt.title("Missingness heatmap (sample)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missingness_heatmap_sample.png", dpi=150)
    plt.close()


def winsorize_series(s: pd.Series, lower_q: float = 0.005, upper_q: float = 0.995) -> pd.Series:
    if s.dtype.kind not in ("i", "u", "f"):
        return s
    lo, hi = s.quantile([lower_q, upper_q])
    return s.clip(lower=lo, upper=hi)


def basic_outlier_handling(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Light winsorization for heavy-tailed fields
    to_winsor = [c for c in numeric_cols if c.lower().startswith(("ret_", "fwd_ret_", "vol", "dist_")) or c in {"Volume"}]
    df[to_winsor] = df[to_winsor].apply(winsorize_series)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Per-ticker forward/back fill for time series continuity
    df = df.groupby("ticker", observed=True).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
    # For any remaining, use global median (numeric) / mode (categorical)
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].cat.add_categories(["UNKNOWN"]).fillna("UNKNOWN")
            else:
                mode_val = df[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
                df[col] = df[col].fillna(fill_val)
    return df


def dataset_overview(df: pd.DataFrame) -> None:
    overview = {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "num_tickers": df["ticker"].nunique(),
    }
    pd.Series(overview).to_csv(OUTPUT_DIR / "overview.csv")

    # Per-ticker coverage
    cov = df.groupby("ticker", observed=True).agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        num_days=("date", "count"),
    ).sort_values("num_days", ascending=False)
    cov.to_csv(OUTPUT_DIR / "ticker_coverage.csv")


def class_imbalance(df: pd.DataFrame) -> None:
    # The provided labels appear historical; still profile them
    label_cols = [c for c in df.columns if c in ("label_5d", "label_20d")]
    for col in label_cols:
        counts = df[col].value_counts(dropna=False)
        counts.to_csv(OUTPUT_DIR / f"class_counts_{col}.csv")

        plt.figure(figsize=(6, 4))
        sns.barplot(x=counts.index.astype(str), y=counts.values)
        plt.title(f"Class distribution: {col}")
        plt.ylabel("count")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"class_distribution_{col}.png", dpi=150)
        plt.close()

        # Pie chart
        plt.figure(figsize=(5, 5))
        plt.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%', startangle=90)
        plt.title(f"Class distribution (pie): {col}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"class_distribution_{col}_pie.png", dpi=150)
        plt.close()


def numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if not c.startswith("fwd_ret_")
    ]
    # Histograms
    sample = df.sample(n=min(200_000, len(df)), random_state=42)
    grid_cols = 4
    for i in range(0, len(numeric_cols), grid_cols * 3):
        cols_chunk = numeric_cols[i:i + grid_cols * 3]
        if not cols_chunk:
            break
        fig, axes = plt.subplots(nrows=int(np.ceil(len(cols_chunk) / grid_cols)), ncols=grid_cols, figsize=(16, 9))
        axes = np.ravel(axes)
        for ax, col in zip(axes, cols_chunk):
            sns.histplot(sample[col], kde=False, bins=50, ax=ax)
            ax.set_title(col)
        for ax in axes[len(cols_chunk):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"histograms_{i//(grid_cols*3)+1}.png", dpi=150)
        plt.close(fig)

    # Boxplots for a subset of key features
    key_cols = [c for c in ["ret_1d", "ret_5d", "ret_20d", "rsi14", "vol_z20", "dist_max20", "dist_min20", "Volume"] if c in df.columns]
    if key_cols:
        fig, axes = plt.subplots(nrows=1, ncols=len(key_cols), figsize=(4*len(key_cols), 4))
        if len(key_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, key_cols):
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(col)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "boxplots_key_features.png", dpi=150)
        plt.close(fig)


def correlation_analysis(df: pd.DataFrame) -> None:
    # Pearson correlation among numeric features (sample to manage size)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    sample = numeric_df.sample(n=min(100_000, len(numeric_df)), random_state=42)
    corr = sample.corr(numeric_only=True)
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Correlation heatmap (sample)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Crosstab correlation between label_5d and label_20d (co-occurrence heatmap)
    if set(["label_5d", "label_20d"]).issubset(df.columns):
        ct = pd.crosstab(df["label_5d"].astype(str), df["label_20d"].astype(str))
        plt.figure(figsize=(6, 5))
        sns.heatmap(ct, annot=True, fmt='d', cmap="Blues")
        plt.title("Co-occurrence: label_5d vs label_20d")
        plt.ylabel("label_5d")
        plt.xlabel("label_20d")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "label5_vs_label20_heatmap.png", dpi=150)
        plt.close()


def ticker_recent_slice(df: pd.DataFrame, recent_years: int = 10) -> pd.DataFrame:
    # Focus EDA on more recent/relevant history
    cutoff = df["date"].max() - pd.DateOffset(years=recent_years)
    return df[df["date"] >= cutoff].copy()


def per_ticker_label_bars(df: pd.DataFrame) -> None:
    if "label_5d" not in df.columns:
        return
    counts = (df.groupby(["ticker", "label_5d"], observed=True).size().unstack(fill_value=0))
    counts = counts[[c for c in ["Buy", "Hold", "Sell"] if c in counts.columns]]
    counts_pct = counts.div(counts.sum(axis=1), axis=0)

    plt.figure(figsize=(9, 5))
    counts_pct.plot(kind="bar", stacked=True, ax=plt.gca(), color=["#22c55e", "#6b7280", "#ef4444"])  # green/gray/red
    plt.title("Label distribution by ticker (label_5d)")
    plt.ylabel("proportion")
    plt.xlabel("ticker")
    plt.legend(title="label_5d")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label5_by_ticker_stacked.png", dpi=150)
    plt.close()


def by_label_box_violin(df: pd.DataFrame) -> None:
    label = "label_5d" if "label_5d" in df.columns else None
    if not label:
        return
    features = [c for c in ["ret_1d", "ret_5d", "ret_20d", "rsi14", "vol_z20"] if c in df.columns]
    if not features:
        return
    # Boxplots
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, features):
        sns.boxplot(data=df, x=label, y=col, ax=ax, order=["Buy","Hold","Sell"] if "Buy" in df[label].unique() else None)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "boxplots_by_label5.png", dpi=150)
    plt.close(fig)

    # Violin plots
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, features):
        sns.violinplot(data=df, x=label, y=col, ax=ax, inner="box", cut=0, order=["Buy","Hold","Sell"] if "Buy" in df[label].unique() else None)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "violins_by_label5.png", dpi=150)
    plt.close(fig)


def rolling_class_distribution(df: pd.DataFrame) -> None:
    if "label_5d" not in df.columns:
        return
    dfm = df.set_index("date")
    monthly = dfm.groupby([pd.Grouper(freq="M"), "label_5d"]).size().unstack(fill_value=0)
    monthly_prop = monthly.div(monthly.sum(axis=1), axis=0)
    monthly_prop.to_csv(OUTPUT_DIR / "rolling_class_distribution_label5_monthly.csv")

    plt.figure(figsize=(10, 4))
    monthly_prop.plot(kind="area", stacked=True, ax=plt.gca(), color=["#22c55e", "#6b7280", "#ef4444"])  # Buy/Hold/Sell colors
    plt.title("Rolling class distribution (monthly, label_5d)")
    plt.ylabel("proportion")
    plt.xlabel("date")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rolling_class_distribution_label5_monthly.png", dpi=150)
    plt.close()


def label_transition_matrix(df: pd.DataFrame) -> None:
    if "label_5d" not in df.columns:
        return
    def transitions(g: pd.DataFrame) -> pd.DataFrame:
        s = g.sort_values("date")["label_5d"].astype(str)
        return pd.DataFrame({"prev": s.shift(1), "curr": s})
    trans = df.groupby("ticker", observed=True).apply(transitions).reset_index(level=0, drop=True)
    trans = trans.dropna()
    mat = pd.crosstab(trans["prev"], trans["curr"]).reindex(index=["Buy","Hold","Sell"], columns=["Buy","Hold","Sell"], fill_value=0)
    mat.to_csv(OUTPUT_DIR / "label5_transition_matrix.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt='d', cmap="Purples")
    plt.title("Label transition matrix (consecutive days, label_5d)")
    plt.ylabel("previous")
    plt.xlabel("current")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label5_transition_matrix.png", dpi=150)
    plt.close()


def mutual_information_topk(df: pd.DataFrame, top_k: int = 30) -> None:
    label_col = "label_5d" if "label_5d" in df.columns else None
    if not label_col:
        return
    y = df[label_col].astype("category").cat.codes
    num = df.select_dtypes(include=[np.number]).copy()
    # Drop targets and forward returns
    drop_cols = [c for c in num.columns if c.startswith("fwd_ret_")]
    num = num.drop(columns=drop_cols, errors="ignore")
    # Clean NaN/inf and constants
    num = num.replace([np.inf, -np.inf], np.nan)
    num = num.fillna(num.median(numeric_only=True))
    num = num.fillna(0)
    # Drop constant columns (no information)
    nunq = num.nunique(dropna=False)
    num = num.loc[:, nunq > 1]
    if num.shape[1] == 0:
        return
    mi = mutual_info_classif(num.values, y.values, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi, index=num.columns).sort_values(ascending=False)
    mi_series.to_csv(OUTPUT_DIR / "mutual_information_label5.csv")

    top = mi_series.head(top_k)
    plt.figure(figsize=(8, max(4, int(0.35*len(top)))))
    sns.barplot(x=top.values, y=top.index, orient="h")
    plt.title("Top features by mutual information with label_5d")
    plt.xlabel("mutual information")
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mi_topk_label5.png", dpi=150)
    plt.close()


def pca_and_vif(df: pd.DataFrame, max_vif_features: int = 30) -> None:
    num = df.select_dtypes(include=[np.number]).copy()
    # Remove obvious targets
    drop_cols = [c for c in num.columns if c.startswith("fwd_ret_")]
    num = num.drop(columns=drop_cols, errors="ignore")
    num = num.replace([np.inf, -np.inf], np.nan)
    num = num.fillna(num.median(numeric_only=True)).fillna(0)
    # Drop constants
    num = num.loc[:, num.nunique(dropna=False) > 1]
    if num.empty:
        return
    scaler = StandardScaler()
    X = scaler.fit_transform(num.values)
    pca = PCA(n_components=min(20, X.shape[1]))
    pca.fit(X)
    var = pd.DataFrame({
        "component": np.arange(1, pca.n_components_+1),
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    var.to_csv(OUTPUT_DIR / "pca_variance.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(var["component"], var["explained_variance_ratio"], marker="o")
    plt.title("PCA explained variance ratio")
    plt.xlabel("component")
    plt.ylabel("variance ratio")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_variance.png", dpi=150)
    plt.close()

    # VIF on subset of features to keep it tractable
    cols = num.columns[:max_vif_features]
    Xv = pd.DataFrame(scaler.fit_transform(num[cols]), columns=cols)
    vif_rows = []
    for i, c in enumerate(cols):
        try:
            vif_rows.append({"feature": c, "vif": float(variance_inflation_factor(Xv.values, i))})
        except Exception:
            vif_rows.append({"feature": c, "vif": np.nan})
    vif = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    vif.to_csv(OUTPUT_DIR / "vif_table.csv", index=False)
    try:
        vif.to_json(OUTPUT_DIR / "vif_table.json", orient="records")
    except Exception:
        pass


def acf_pacf_and_rolling_vol(df: pd.DataFrame, max_lag: int = 20) -> None:
    if "ret_1d" not in df.columns:
        return
    for ticker, g in df.groupby("ticker", observed=True):
        s = g.sort_values("date")["ret_1d"].dropna()
        if len(s) < max_lag + 5:
            continue
        ac = sm_acf(s, nlags=max_lag, fft=True)
        pc = sm_pacf(s, nlags=max_lag, method='yw')

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].stem(range(len(ac)), ac, basefmt=" ")
        axes[0].set_title(f"ACF {ticker}")
        axes[1].stem(range(len(pc)), pc, basefmt=" ")
        axes[1].set_title(f"PACF {ticker}")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"acf_pacf_{ticker}.png", dpi=150)
        plt.close(fig)

        # Rolling volatility
        rv = s.rolling(20).std()
        plt.figure(figsize=(8, 3))
        plt.plot(g.sort_values("date")["date"], rv)
        plt.title(f"Rolling 20D volatility {ticker}")
        plt.xlabel("date")
        plt.ylabel("rolling std of ret_1d")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"rolling_vol_{ticker}.png", dpi=150)
        plt.close()


def leakage_audit(df: pd.DataFrame, lags: range = range(-10, 11)) -> None:
    # Correlate numeric features shifted by lag with forward returns
    targ = "fwd_ret_5d" if "fwd_ret_5d" in df.columns else None
    if not targ:
        return
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {targ}]
    results = []
    for col in num_cols:
        row = {"feature": col}
        for lag in lags:
            vals = []
            for _, g in df.groupby("ticker", observed=True):
                x = g[col].shift(lag)
                y = g[targ]
                v = pd.concat([x, y], axis=1).dropna()
                if len(v) > 30:
                    vals.append(v[col].corr(v[targ]))
            row[f"lag_{lag}"] = np.nanmean(vals) if vals else np.nan
        results.append(row)
    audit = pd.DataFrame(results)
    audit.to_csv(OUTPUT_DIR / "leakage_audit_corr_vs_lag.csv", index=False)

    # Heatmap of absolute correlations for top suspicious features
    audit_abs = audit.set_index("feature").abs()
    suspicious = audit_abs.max(axis=1).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    sns.heatmap(audit.loc[audit["feature"].isin(suspicious.index)].set_index("feature"), cmap="coolwarm", center=0)
    plt.title("Leakage audit: corr(feature_t+lag, fwd_ret_5d)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "leakage_audit_heatmap.png", dpi=150)
    plt.close()


def price_lines_recent(df: pd.DataFrame) -> None:
    # Plot normalized Close price per ticker over the recent window
    if not {"date", "ticker", "Close"}.issubset(df.columns):
        return
    recent_df = df.copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker, g in recent_df.groupby("ticker", observed=True):
        g = g.sort_values("date")
        if g["Close"].notna().sum() < 5:
            continue
        base = g["Close"].iloc[0]
        if base == 0 or pd.isna(base):
            continue
        series = (g["Close"] / base) * 100.0
        ax.plot(g["date"], series, label=str(ticker))
    ax.set_title("Price change (normalized to 100 at start) — last 10 years")
    ax.set_ylabel("index (start=100)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "price_lines_last10y.png", dpi=150)
    plt.close(fig)


def profile_with_ydata(df: pd.DataFrame) -> None:
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df.sample(n=min(100000, len(df)), random_state=42), title="EDA Profile (sample)", minimal=True)
        profile.to_file(OUTPUT_DIR / "ydata_profile.html")
    except Exception as e:
        # Do not fail EDA if profiling struggles; log instead
        (OUTPUT_DIR / "ydata_profile_error.txt").write_text(str(e))


def main() -> None:
    ensure_output_dir()
    df = load_data(DATA_PATH)
    df = standardize_formats(df)
    df = remove_duplicates(df)
    summarize_missingness(df)

    df_recent = ticker_recent_slice(df, recent_years=10)

    df_recent = basic_outlier_handling(df_recent)
    df_recent = impute_missing(df_recent)

    dataset_overview(df_recent)
    price_lines_recent(df_recent)
    class_imbalance(df_recent)
    per_ticker_label_bars(df_recent)
    numeric_distributions(df_recent)
    correlation_analysis(df_recent)
    by_label_box_violin(df_recent)
    rolling_class_distribution(df_recent)
    label_transition_matrix(df_recent)
    mutual_information_topk(df_recent)
    pca_and_vif(df_recent)
    acf_pacf_and_rolling_vol(df_recent)
    leakage_audit(df_recent)
    profile_with_ydata(df_recent)

    # Save a typed schema snapshot
    schema = pd.DataFrame({
        "column": df_recent.columns,
        "dtype": [str(t) for t in df_recent.dtypes]
    })
    schema.to_csv(OUTPUT_DIR / "schema_dtypes.csv", index=False)


if __name__ == "__main__":
    main()


