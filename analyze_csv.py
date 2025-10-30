import pandas as pd

df = pd.read_csv('panel_daily_with_msq.csv')
print(f"📊 Total columns: {len(df.columns)}")
print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
print(f"🎯 Unique tickers: {df['ticker'].nunique()}")

print("\n" + "="*60)
print("NEW MSQ FUNDAMENTAL COLUMNS (33 total):")
print("="*60)

msq_cols = [col for col in df.columns if col.startswith('msq_')]
for i, col in enumerate(msq_cols, 1):
    missing_pct = (df[col].isna().sum() / len(df)) * 100
    print(f"{i:2d}. {col:<35} | Missing: {missing_pct:>5.1f}%")

print("\n" + "="*60)
print("OTHER CHANGES:")
print("="*60)
print("• 'prediction_20d' added (replaces 'label_20d' as a prediction column)")
print("• 'label_5d' retained as target variable")

print("\n" + "="*60)
print("SUMMARY OF MSQ METRICS:")
print("="*60)
print("📈 Valuation: pb, peTTM, pfcfTTM, psTTM, ptbv, ev, evEbitdaTTM, evRevenueTTM")
print("💰 Profitability: eps, ebitPerShare, netMargin, grossMargin, operatingMargin, pretaxMargin")
print("📊 Returns: roaTTM, roeTTM, roicTTM, rotcTTM")
print("💵 Cash: fcfMargin, fcfPerShareTTM, cashRatio")
print("🔢 Leverage: debt ratios (total, longterm, net)")
print("⚖️ Liquidity: currentRatio, quickRatio")
print("🔄 Efficiency: assetTurnover, inventoryTurnover, receivablesTurnover")

