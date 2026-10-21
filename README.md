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

## 2. RFM analysis

[`notebooks/02_rfm_analysis.ipynb`](notebooks/02_rfm_analysis.ipynb)

Each customer is described by **Recency** (days since last purchase), **Frequency** (number of
invoices) and **Monetary** (net spend), measured at 2011-12-10. Revenue is highly concentrated:
the top 20% of customers generate **77%** of net revenue, the top 1% generate 31%.

- Recency and monetary are scored 1–5 by quintiles.
- Frequency uses fixed thresholds (1 / 2 / 3–4 / 5–9 / 10+ invoices): 27.6% of customers have a
  single invoice, so quintiles either collapse or give identical customers different scores.
- Segments come from the standard recency × frequency grid. Monetary is left out of the grid
  because it is strongly correlated with frequency (Spearman 0.86) and is used to describe segment value.
- The RFM function takes the snapshot date as a parameter, so the churn model can recompute it at
  a past cutoff without leakage.

| Segment | Customers | % customers | % net revenue | Recommended action |
|---|---:|---:|---:|---|
| Champions | 822 | 14.1% | 52.4% | Reward: early access, loyalty perks, no discounts needed |
| Loyal | 1,027 | 17.6% | 26.8% | Upsell and cross-sell, ask for referrals |
| Potential loyalists | 715 | 12.2% | 4.2% | Build the habit: volume offers, pre-Christmas reminders |
| New customers | 77 | 1.3% | 0.2% | Onboarding and a reason to place a second order |
| Promising | 161 | 2.8% | 0.3% | Personalised recommendations |
| Need attention | 281 | 4.8% | 2.0% | Time-limited offer based on past purchases |
| About to sleep | 421 | 7.2% | 1.6% | Reactivation email before they lapse |
| Can't lose | 57 | 1.0% | 2.7% | Top-priority win-back with personal contact |
| At risk | 646 | 11.1% | 5.4% | Win-back campaign with a real incentive |
| Hibernating | 1,632 | 27.9% | 4.3% | Low-cost automated campaigns only |

![RFM segments: size vs revenue](reports/figures/02_rfm_segments.png)

## 3. Segmentation with clustering

[`notebooks/03_segmentation.ipynb`](notebooks/03_segmentation.ipynb)

K-means on the actual recency, frequency and monetary values, log-transformed (skewness of monetary
drops from 26.8 to 0.1) and standardised so that days, invoices and pounds weigh equally.

- **k = 4**: second-best silhouette after the trivial k = 2 (0.36 vs 0.34 for k = 3) and almost
  identical clusters across random seeds (adjusted Rand index 0.99).
- **Gaussian mixture rejected**: because frequency is an integer, the BIC rewards components squeezed
  onto single values ("exactly 2 invoices") and keeps falling up to k = 9. With 4 components it agrees
  only partly with k-means (ARI 0.40) and is uncertain about 29% of customers.

| Cluster | % customers | Days since last purchase | Invoices | Net spend | % revenue | Recommended action |
|---|---:|---:|---:|---:|---:|---|
| Best customers | 20.8% | 17 | 13 | £4,711 | 74.1% | Protect: account management, early access, watch for slowdowns |
| Fading regulars | 26.0% | 188 | 4 | £1,342 | 16.4% | Biggest opportunity: personalised win-back, trigger when the usual gap is exceeded |
| Recent, low frequency | 20.0% | 24 | 3 | £725 | 5.9% | Nurture towards a habit: recommendations, bundles, pre-Christmas reminders |
| Lost one-timers | 33.3% | 398 | 1 | £254 | 3.6% | Low priority: cheap automated reactivation only |

*Medians per cluster.*

![Median profile of each cluster](reports/figures/03_cluster_profiles.png)

**Comparison with RFM segments:** the two methods largely agree (83% of Champions are Best customers,
92% of Hibernating are Lost one-timers). The exception is the **Loyal** segment, which k-means splits
into heavy buyers still active (median 36 days, 12 invoices) and moderate buyers whose last order is
already three months old (89 days, 6 invoices): part of a segment that looks healthy in the RFM grid
is already losing the habit.

![RFM segments vs clusters](reports/figures/03_rfm_vs_clusters.png)

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
