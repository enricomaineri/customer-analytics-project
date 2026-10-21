"""Customer Lifetime Value with BG/NBD (number of purchases) and Gamma-Gamma (value per purchase).

The `lifetimes` models are fitted on a per-customer summary:
- frequency: number of repeat purchase days (the first purchase is not counted)
- recency:   customer age at the last purchase (first to last purchase), in weeks
- T:         customer age at the end of the period (first purchase to end), in weeks
- monetary_value: average value of the repeat purchase days

Recency here is NOT the RFM recency (days since the last purchase).
"""

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data

DAYS_PER_WEEK = 7


def daily_purchases(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per customer and purchase day, with the revenue of that day.

    BG/NBD counts purchase occasions, so several invoices on the same day count once.
    """
    return (transactions
            .assign(day=transactions["invoice_date"].dt.normalize())
            .groupby(["customer_id", "day"], as_index=False)["revenue"].sum())


def _days_to_weeks(summary: pd.DataFrame, columns) -> pd.DataFrame:
    summary = summary.copy()
    summary[columns] = summary[columns] / DAYS_PER_WEEK
    return summary


def customer_summary(daily: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Frequency, recency, T (weeks) and monetary value per customer up to `end`."""
    summary = summary_data_from_transaction_data(
        daily, "customer_id", "day", monetary_value_col="revenue", observation_period_end=end
    )
    return _days_to_weeks(summary, ["recency", "T"])


def calibration_holdout(daily: pd.DataFrame, calibration_end: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Summary on the calibration period plus what each customer did in the holdout period.

    Only customers whose first purchase is in the calibration period are included.
    """
    summary = calibration_and_holdout_data(
        daily, "customer_id", "day",
        calibration_period_end=calibration_end, observation_period_end=end,
        monetary_value_col="revenue",
    )
    return _days_to_weeks(summary, ["recency_cal", "T_cal", "duration_holdout"])


def fit_bgnbd(frequency, recency, T, model=BetaGeoFitter, penalizer: float = 0.0):
    """Fit a BG/NBD-type model (BetaGeoFitter or ModifiedBetaGeoFitter)."""
    return model(penalizer_coef=penalizer).fit(frequency, recency, T)


def fit_gamma_gamma(frequency, monetary_value, penalizer: float = 0.0) -> GammaGammaFitter:
    """Fit Gamma-Gamma on repeat customers only (it needs at least one repeat purchase)."""
    repeat = frequency > 0
    return GammaGammaFitter(penalizer_coef=penalizer).fit(frequency[repeat], monetary_value[repeat])


def naive_purchases(frequency, T, horizon_weeks: float):
    """Baseline: each customer keeps the repeat-purchase rate observed so far."""
    return frequency / T.clip(lower=1 / DAYS_PER_WEEK) * horizon_weeks
