# E-Commerce Profitability Analysis

A Power BI dashboard that analyses e-commerce profitability by product category and marketing channel for January to December 2024.

## Key Metrics (2024)

| Metric | Value |
|---|---|
| Total Revenue | ₹12,48,930 |
| Total Cost | ₹8,26,540 |
| Total Profit | ₹4,22,390 |
| Profit Margin | 33.8% |

## Dashboard Contents

- KPI cards: revenue, cost, profit and profit margin, each compared with the previous period
- Profit by product category
- Revenue vs profit monthly trend
- Profit by marketing channel
- Revenue share by product category
- Profit share by marketing channel
- Category and channel wise profitability table
- Filters: date, product category, marketing channel, region and product name

## Project Files

- `data/ecommerce_orders.csv`: order-level dataset (2023 and 2024)
- `powerbi/measures.dax`: DAX measures and date table
- `powerbi/theme.json`: dashboard colour theme
- `scripts/generate_data.py`: script used to create the sample dataset

## Tools Used

Power BI, DAX, Python (pandas, NumPy, SciPy)

## Data Note

The dataset is synthetic, created for this project. It does not represent a real business.
