"""RFM (Recency, Frequency, Monetary) table and rule-based segments.

Everything is computed relative to a snapshot date and uses only data before
it, so the same functions can build churn features at a past cutoff without
looking into the future.
"""

import pandas as pd

# Fixed frequency thresholds: 1 / 2 / 3-4 / 5-9 / 10+ invoices. Quintiles do not
# work here because more than a quarter of the customers have a single invoice.
FREQUENCY_BINS = [0, 1, 2, 4, 9, float("inf")]

# Standard RFM segment map on the recency and frequency scores (1 = worst, 5 = best).
# Every one of the 25 (R, F) combinations falls in exactly one segment.
SEGMENT_RULES = [
    # (recency scores, frequency scores, segment)
    ((5,), (4, 5), "Champions"),
    ((3, 4), (4, 5), "Loyal"),
    ((4, 5), (2, 3), "Potential loyalists"),
    ((5,), (1,), "New customers"),
    ((4,), (1,), "Promising"),
    ((3,), (3,), "Need attention"),
    ((3,), (1, 2), "About to sleep"),
    ((1, 2), (5,), "Can't lose"),
    ((1, 2), (3, 4), "At risk"),
    ((1, 2), (1, 2), "Hibernating"),
]
SEGMENT_ORDER = [name for _, _, name in SEGMENT_RULES]


def compute_rfm(
    transactions: pd.DataFrame, returns: pd.DataFrame, snapshot_date: pd.Timestamp
) -> pd.DataFrame:
    """Recency (days), frequency (invoices) and net monetary value per customer.

    Only lines strictly before `snapshot_date` are used. Customers without a
    purchase before the snapshot are not included, even if they have returns.
    """
    tx = transactions[transactions["invoice_date"] < snapshot_date]
    rt = returns[returns["invoice_date"] < snapshot_date]
    by_customer = tx.groupby("customer_id")

    # Compare calendar days, so a purchase on the day before the snapshot has recency 1
    last_purchase_day = by_customer["invoice_date"].max().dt.normalize()
    returned = rt.groupby("customer_id")["revenue"].sum()

    rfm = pd.DataFrame({
        "recency": (snapshot_date.normalize() - last_purchase_day).dt.days,
        "frequency": by_customer["invoice"].nunique(),
        "monetary": by_customer["revenue"].sum() + returned.reindex(last_purchase_day.index, fill_value=0),
    })
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add 1-5 scores for each dimension, where 5 is always the best customer."""
    scored = rfm.copy()
    # Low recency is good, so the labels are reversed
    scored["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    scored["f_score"] = pd.cut(rfm["frequency"], FREQUENCY_BINS, labels=[1, 2, 3, 4, 5]).astype(int)
    scored["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return scored


def assign_segments(scored: pd.DataFrame) -> pd.Series:
    """Map each customer's (R, F) scores to a named segment."""
    lookup = {
        (r, f): name
        for recency_scores, frequency_scores, name in SEGMENT_RULES
        for r in recency_scores
        for f in frequency_scores
    }
    segments = [lookup[(r, f)] for r, f in zip(scored["r_score"], scored["f_score"])]
    return pd.Series(
        pd.Categorical(segments, categories=SEGMENT_ORDER), index=scored.index, name="segment"
    )
