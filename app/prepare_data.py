"""Build the single customer table used by the dashboard.

Run from the project root after the five notebooks:

    python app/prepare_data.py

The output is small and versioned, so the dashboard can be deployed without the
raw data. Profiles, segments and CLV are as of 2011-12-10; churn risk comes from
the 2011-09-10 test snapshot of notebook 05, together with what actually happened.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT = Path(__file__).resolve().parent / "data" / "customers.parquet"


def main():
    segments = pd.read_parquet(PROCESSED_DIR / "segments.parquet")
    clv = pd.read_parquet(PROCESSED_DIR / "clv.parquet")
    churn = pd.read_parquet(PROCESSED_DIR / "churn_scores.parquet")
    transactions = pd.read_parquet(PROCESSED_DIR / "transactions.parquet")

    customers = segments[["recency", "frequency", "monetary", "r_score", "f_score", "m_score", "segment", "cluster"]]
    customers = customers.join(clv[["p_alive", "expected_purchases_12m", "clv_12m"]])
    customers["repeat_purchase_days"] = clv["frequency"]
    customers["country"] = transactions.groupby("customer_id")["country"].first()

    # Customers who did not buy in the 9 months before the September cutoff are not in the snapshot
    customers = customers.join(churn[["churn_risk", "churn", "monetary", "revenue_at_risk"]].rename(columns={
        "churn_risk": "churn_risk_sep_2011",
        "churn": "churned_after_sep_2011",
        "monetary": "spend_9m_before_sep_2011",
    }))

    OUTPUT.parent.mkdir(exist_ok=True)
    customers.reset_index().to_parquet(OUTPUT, index=False)
    print(f"Saved {len(customers):,} customers to {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
