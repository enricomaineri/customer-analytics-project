# Customer Analytics on Online Retail II

End-to-end customer analytics on two years of transactions from a UK-based online
gift retailer: data cleaning, RFM analysis, customer segmentation, Customer Lifetime
Value (BG/NBD + Gamma-Gamma) and churn prediction.

> Work in progress.

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI Machine
Learning Repository, CC BY 4.0): ~1.07M invoice lines, Dec 2009 – Dec 2011.

The data is not versioned. Download the zip from UCI and place `online_retail_II.xlsx`
in `data/raw/`.

## 1. Data cleaning

[`notebooks/01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb)

| Step | Rows left | Rows removed | Customers |
|---|---:|---:|---:|
| Raw data (both sheets) | 1,067,371 | – | 5,942 |
| 1. Remove overlap between the two sheets | 1,044,848 | 22,523 | 5,942 |
| 2. Drop exact duplicate rows | 1,033,036 | 11,812 | 5,942 |
| 3. Drop lines without customer ID | 797,885 | 235,151 | 5,942 |
| 4. Drop non-positive prices | 797,815 | 70 | 5,939 |
| 5. Drop non-product codes (postage, fees, adjustments) | 794,183 | 3,632 | 5,875 |
| 6–7. Remove returns paired with the purchase they cancel | 782,137 | 12,046 | 5,862 |

Key decisions:

- **Missing customer IDs (22.8% of lines, 15% of revenue) are dropped.** IDs are missing for
  whole invoices, never for single lines, so they cannot be imputed. All customer metrics
  describe registered customers.
- **Returns are paired with the purchase they cancel and both are removed** when customer,
  product, price and quantity match exactly. Only 34% of return lines pair up, but they account
  for 77% of the returned value (large orders cancelled within minutes). Without this step a
  cancelled order would still count as a transaction in the CLV models.
- **Unpaired returns** (mostly partial) are kept in a separate table, used later for net spend
  and as a churn feature.
- **Large wholesale orders are kept**: they are real behaviour, not errors.

Result: 5,839 customers with 36,337 purchase invoices between Dec 2009 and Dec 2011.

![Monthly revenue and active customers](reports/figures/01_monthly_activity.png)

## Project structure

```
├── app/              # Streamlit dashboard (optional)
├── data/
│   ├── raw/          # original UCI file (not versioned)
│   └── processed/    # outputs of each notebook (not versioned)
├── notebooks/        # one notebook per phase, run in order
├── reports/figures/  # charts used in this README
├── src/              # helper functions shared across notebooks
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
