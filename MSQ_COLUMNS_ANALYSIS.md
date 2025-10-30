# Analysis of New MSQ Columns in panel_daily_with_msq.csv

## Summary

Your CSV now contains **33 new MSQ fundamental analysis columns** in addition to the existing price, technical, and sentiment features.

## New MSQ Columns (33 total)

### Valuation Metrics (8)
1. `msq_pb` - Price-to-Book ratio
2. `msq_peTTM` - Price-to-Earnings (Trailing 12 Months)
3. `msq_pfcfTTM` - Price-to-Free Cash Flow (TTM)
4. `msq_psTTM` - Price-to-Sales (TTM)
5. `msq_ptbv` - Price-to-Tangible Book Value
6. `msq_ev` - Enterprise Value
7. `msq_evEbitdaTTM` - EV/EBITDA (TTM)
8. `msq_evRevenueTTM` - EV/Revenue (TTM)

### Profitability Metrics (6)
9. `msq_eps` - Earnings Per Share
10. `msq_ebitPerShare` - EBIT Per Share
11. `msq_netMargin` - Net Profit Margin
12. `msq_grossMargin` - Gross Profit Margin
13. `msq_operatingMargin` - Operating Profit Margin
14. `msq_pretaxMargin` - Pretax Profit Margin

### Return Metrics (4)
15. `msq_roaTTM` - Return on Assets (TTM)
16. `msq_roeTTM` - Return on Equity (TTM)
17. `msq_roicTTM` - Return on Invested Capital (TTM)
18. `msq_rotcTTM` - Return on Total Capital (TTM)

### Cash Flow Metrics (3)
19. `msq_fcfMargin` - Free Cash Flow Margin
20. `msq_fcfPerShareTTM` - Free Cash Flow Per Share (TTM)
21. `msq_cashRatio` - Cash Ratio

### Leverage & Debt Metrics (6)
22. `msq_longtermDebtTotalAsset` - Long-term Debt / Total Assets
23. `msq_longtermDebtTotalCapital` - Long-term Debt / Total Capital
24. `msq_longtermDebtTotalEquity` - Long-term Debt / Total Equity
25. `msq_netDebtToTotalCapital` - Net Debt / Total Capital
26. `msq_netDebtToTotalEquity` - Net Debt / Total Equity
27. `msq_totalDebtToEquity` - Total Debt / Equity
28. `msq_totalDebtToTotalAsset` - Total Debt / Total Assets
29. `msq_totalDebtToTotalCapital` - Total Debt / Total Capital
30. `msq_totalRatio` - Total Ratio

### Liquidity Metrics (2)
31. `msq_currentRatio` - Current Ratio
32. `msq_quickRatio` - Quick Ratio

### Efficiency Metrics (4)
33. `msq_assetTurnoverTTM` - Asset Turnover (TTM)
34. `msq_inventoryTurnoverTTM` - Inventory Turnover (TTM)
35. `msq_receivablesTurnoverTTM` - Receivables Turnover (TTM)
36. `msq_sgaToSale` - SG&A / Sales

### Other Metrics (2)
37. `msq_bookValue` - Book Value
38. `msq_tangibleBookValue` - Tangible Book Value
39. `msq_salesPerShare` - Sales Per Share
40. `msq_payoutRatioTTM` - Payout Ratio (TTM)

## Other Changes

### New Column
- **`prediction_20d`**: Appears to be model predictions for the 20-day horizon (Buy/Hold/Sell)

### Existing Columns
- **`label_5d`**: Still present as the target variable for 5-day classification

## Total Column Count

The CSV now has **62 columns total** (up from ~29 in the original):
- Original: date, ticker, OHLCV, sentiment, returns, technical indicators, labels
- New: 33 MSQ fundamental metrics

## Impact on ML Pipeline

### Feature Engineering
- These MSQ columns will be powerful features for fundamental analysis
- Many may have high missingness (fundamentals reported quarterly, not daily)
- Consider:
  - Forward-filling fundamental data within quarters
  - Creating "rolling" fundamental features (trailing averages)
  - Imputing missing values carefully (may indicate data quality issues)

### Model Performance
- Fundamanetal features should improve prediction quality significantly
- Combine with technical indicators for comprehensive stock analysis
- Watch for multicollinearity among related ratios

### Next Steps
1. Run EDA on new CSV to assess missingness patterns
2. Implement robust imputation for fundamental metrics
3. Add feature engineering for MSQ columns
4. Train models with expanded feature set

