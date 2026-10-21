"""Churn dataset built at a cutoff date, using only what was known at that date.

Churn = no purchase in the HORIZON_DAYS after the cutoff. Features come from the
LOOKBACK_DAYS before the cutoff, with returns paired to purchases again using only
the returns that had already happened (phase 1 paired them on the whole period).
"""

import numpy as np
import pandas as pd

from src import cleaning

LOOKBACK_DAYS = 273   # 9 months: the longest history available at the first cutoff
HORIZON_DAYS = 91     # 3 months: from mid September to the end of the Christmas season

# Count-like features with a long right tail, log-transformed before linear models
SKEWED_FEATURES = [
    "recency", "tenure", "frequency", "monetary", "avg_order_value",
    "invoices_30d", "revenue_30d", "invoices_90d", "revenue_90d",
    "distinct_products", "mean_days_between", "overdue_ratio",
]
OTHER_FEATURES = ["return_rate", "is_uk", "one_time_buyer"]
FEATURES = SKEWED_FEATURES + OTHER_FEATURES


def lines_known_at(lines: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Purchases and returns in the lookback window, paired only with returns before the cutoff."""
    start = cutoff - pd.Timedelta(days=LOOKBACK_DAYS)
    window = lines[(lines["invoice_date"] >= start) & (lines["invoice_date"] < cutoff)]
    purchases, returns = cleaning.split_purchases_returns(window)
    purchases, returns, _ = cleaning.cancel_matched_returns(purchases, returns)
    return purchases, returns


def build_features(purchases: pd.DataFrame, returns: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per customer who bought in the lookback window."""
    invoices = (purchases
                .groupby(["customer_id", "invoice"], as_index=False)
                .agg(date=("invoice_date", "min"), value=("revenue", "sum")))
    by_customer = invoices.groupby("customer_id")

    features = pd.DataFrame({
        "recency": (cutoff - by_customer["date"].max()).dt.days,
        "tenure": (cutoff - by_customer["date"].min()).dt.days,
        "frequency": by_customer["invoice"].nunique(),
        "monetary": by_customer["value"].sum(),
        "avg_order_value": by_customer["value"].mean(),
    })

    # Recent activity: a slowdown shows up here before it shows up in recency
    for days in (30, 90):
        recent = invoices[invoices["date"] >= cutoff - pd.Timedelta(days=days)].groupby("customer_id")
        features[f"invoices_{days}d"] = recent["invoice"].nunique().reindex(features.index, fill_value=0)
        features[f"revenue_{days}d"] = recent["value"].sum().reindex(features.index, fill_value=0)

    # Usual rhythm of each customer, and how late they are compared with it
    purchase_days = (invoices.assign(day=invoices["date"].dt.normalize())
                     .drop_duplicates(["customer_id", "day"])
                     .sort_values("day"))
    gaps = purchase_days.groupby("customer_id")["day"].diff().dt.days
    features["mean_days_between"] = gaps.groupby(purchase_days["customer_id"]).mean().reindex(features.index)
    features["overdue_ratio"] = features["recency"] / features["mean_days_between"]

    features["distinct_products"] = purchases.groupby("customer_id")["stock_code"].nunique()
    # Returns of purchases made before the window can exceed the window's purchases, hence the cap
    returned = -returns.groupby("customer_id")["revenue"].sum()
    features["return_rate"] = (returned.reindex(features.index, fill_value=0) / features["monetary"]).clip(upper=1)
    features["is_uk"] = (purchases.groupby("customer_id")["country"].first() == "United Kingdom").astype(int)
    features["one_time_buyer"] = (features["frequency"] == 1).astype(int)
    return features[FEATURES]


def churn_labels(transactions: pd.DataFrame, customers: pd.Index, cutoff: pd.Timestamp) -> pd.Series:
    """1 if the customer makes no purchase in the HORIZON_DAYS after the cutoff."""
    end = cutoff + pd.Timedelta(days=HORIZON_DAYS)
    future = transactions[(transactions["invoice_date"] >= cutoff) & (transactions["invoice_date"] < end)]
    return pd.Series((~customers.isin(future["customer_id"])).astype(int), index=customers, name="churn")


def build_dataset(lines: pd.DataFrame, transactions: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Features known at the cutoff plus the churn label observed afterwards."""
    purchases, returns = lines_known_at(lines, cutoff)
    assert purchases["invoice_date"].max() < cutoff and returns["invoice_date"].max() < cutoff
    features = build_features(purchases, returns, cutoff)
    return features.join(churn_labels(transactions, features.index, cutoff))


def top_share_rate(y_true: pd.Series, scores, share: float = 0.2) -> float:
    """Churn rate among the `share` of customers with the highest risk score."""
    ranks = pd.Series(np.asarray(scores), index=y_true.index).rank(ascending=False, method="first")
    return y_true[ranks <= len(y_true) * share].mean()
