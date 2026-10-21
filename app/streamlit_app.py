"""Customer analytics dashboard: segments, single-customer view and a campaign simulator.

Run from the project root with:

    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_FILE = Path(__file__).parent / "data" / "customers.parquet"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
RANDOM_STATE = 42

ACTIONS = {
    "Champions": "Reward them: early access to new collections, loyalty perks. No discounts needed",
    "Loyal": "Upsell and cross-sell, ask for reviews and referrals",
    "Potential loyalists": "Build the habit: volume offers, reminders before the Christmas season",
    "New customers": "Onboarding: welcome email, best sellers, a reason to place the second order",
    "Promising": "Light follow-up with personalised recommendations",
    "Need attention": "Time-limited offer based on what they bought before",
    "About to sleep": "Reactivation email before they become hibernating",
    "Can't lose": "Top-priority win-back: personal contact, find out why they stopped",
    "At risk": "Win-back campaign with a meaningful incentive",
    "Hibernating": "Low-cost automated campaigns only",
    "Best customers": "Protect them: account management, early access, watch for any slowdown",
    "Fading regulars": "Personalised win-back based on past purchases; trigger when the usual gap is exceeded",
    "Recent, low frequency": "Nurture towards a habit: recommendations, bundles, pre-Christmas reminders",
    "Lost one-timers": "Low priority: cheap automated reactivation only",
}

EXAMPLES = {
    14911: "buys more than twice a week",
    17850: "155 orders, silent for a year: a 'false friend'",
    13902: "large wholesaler, gone since 2010",
    12367: "one recent order: the model knows little",
}


@st.cache_data
def load_customers() -> pd.DataFrame:
    return pd.read_parquet(DATA_FILE).set_index("customer_id")


def group_summary(customers: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = customers.groupby(column, observed=False).agg(
        customers=("recency", "size"),
        median_recency=("recency", "median"),
        median_invoices=("frequency", "median"),
        median_spend=("monetary", "median"),
        revenue=("monetary", "sum"),
        median_clv=("clv_12m", "median"),
    )
    summary["share_of_customers"] = summary["customers"] / summary["customers"].sum()
    summary["share_of_revenue"] = summary["revenue"] / summary["revenue"].sum()
    summary["recommended_action"] = summary.index.map(ACTIONS)
    return summary.drop(columns="revenue")


def size_vs_revenue_chart(summary: pd.DataFrame, group_label: str) -> alt.Chart:
    long = (summary[["share_of_customers", "share_of_revenue"]]
            .rename(columns={"share_of_customers": "Share of customers", "share_of_revenue": "Share of net revenue"})
            .rename_axis(group_label)
            .reset_index()
            .melt(id_vars=group_label, var_name="Measure", value_name="Share"))
    order = list(summary.index)
    return (alt.Chart(long)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                y=alt.Y(f"{group_label}:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
                yOffset=alt.YOffset("Measure:N"),
                x=alt.X("Share:Q", axis=alt.Axis(format="%", tickCount=6), title=None),
                color=alt.Color("Measure:N", scale=alt.Scale(range=[BLUE, ORANGE]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip(f"{group_label}:N"), alt.Tooltip("Measure:N"),
                         alt.Tooltip("Share:Q", format=".1%")],
            )
            # One step per bar, so the chart grows with the number of groups
            .properties(height=alt.Step(16)))


def campaign_lists(snapshot: pd.DataFrame) -> dict[str, pd.Index]:
    """Order in which each targeting rule would contact customers."""
    shuffled = snapshot.sample(frac=1, random_state=RANDOM_STATE).index
    return {
        "Highest churn risk": snapshot["churn_risk_sep_2011"].sort_values(ascending=False).index,
        "Highest revenue at risk": snapshot["revenue_at_risk"].sort_values(ascending=False).index,
        "Highest past spend": snapshot["spend_9m_before_sep_2011"].sort_values(ascending=False).index,
        "Random": shuffled,
    }


def campaign_result(snapshot: pd.DataFrame, contacted: pd.Index) -> dict:
    reached = snapshot.loc[contacted]
    churners = reached[reached["churned_after_sep_2011"] == 1]
    return {
        "churners_reached": len(churners),
        "churn_rate": churners.shape[0] / len(reached),
        "spend_of_churners": churners["spend_9m_before_sep_2011"].sum(),
    }


st.set_page_config(page_title="Customer Analytics", page_icon="📊", layout="wide")
customers = load_customers()

st.title("Customer analytics · Online Retail II")
st.caption("UK online giftware retailer, registered customers, Dec 2009 – Dec 2011. "
           "Profiles, segments and CLV as of 10 Dec 2011; churn risk as of 10 Sep 2011.")

overview_tab, customer_tab, campaign_tab = st.tabs(["Overview", "Customer lookup", "Campaign simulator"])

with overview_tab:
    ranked = customers["monetary"].sort_values(ascending=False)
    top_share = ranked.head(int(len(ranked) * 0.2)).sum() / ranked.sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(customers):,}")
    k2.metric("Net revenue", f"£{customers['monetary'].sum() / 1e6:.1f}M")
    k3.metric("Revenue from top 20% customers", f"{top_share:.0%}")
    k4.metric("Predicted CLV, next 12 months", f"£{customers['clv_12m'].sum() / 1e6:.1f}M")

    grouping = st.radio("Group customers by", ["RFM segment", "K-means cluster"], horizontal=True)
    column = "segment" if grouping == "RFM segment" else "cluster"
    summary = group_summary(customers, column)
    table = summary.rename_axis(grouping).assign(
        share_of_customers=summary["share_of_customers"] * 100,
        share_of_revenue=summary["share_of_revenue"] * 100,
    )

    st.altair_chart(size_vs_revenue_chart(summary, grouping), width="stretch")
    st.dataframe(
        table,
        width="stretch",
        column_config={
            "customers": st.column_config.NumberColumn("Customers", format="%d"),
            "median_recency": st.column_config.NumberColumn("Days since last purchase", format="%d"),
            "median_invoices": st.column_config.NumberColumn("Invoices", format="%d"),
            "median_spend": st.column_config.NumberColumn("Net spend", format="£%.0f"),
            "median_clv": st.column_config.NumberColumn("CLV 12m", format="£%.0f"),
            "share_of_customers": st.column_config.ProgressColumn("% customers", format="%.1f%%", min_value=0, max_value=100),
            "share_of_revenue": st.column_config.ProgressColumn("% revenue", format="%.1f%%", min_value=0, max_value=100),
            "recommended_action": st.column_config.TextColumn("Recommended action", width="large"),
        },
    )
    st.caption("Medians per group. RFM segments come from fixed rules on 1–5 scores; "
               "clusters from k-means on log-transformed recency, frequency and monetary.")

with customer_tab:
    st.markdown("Some customers worth a look: " + " · ".join(f"**{cid}** ({note})" for cid, note in EXAMPLES.items()))
    ids = customers.index.tolist()
    customer_id = st.selectbox("Customer ID (type to search)", ids, index=ids.index(17850))
    c = customers.loc[customer_id]

    st.subheader(f"Customer {customer_id} · {c['country']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Days since last purchase", f"{c['recency']:,}")
    col2.metric("Invoices", f"{c['frequency']:,}")
    col3.metric("Net spend", f"£{c['monetary']:,.0f}")

    col1, col2, col3 = st.columns(3)
    col1.metric(f"RFM segment (R{c['r_score']} F{c['f_score']} M{c['m_score']})", c["segment"])
    col2.metric("K-means cluster", c["cluster"])
    col3.metric("Predicted CLV, next 12 months", f"£{c['clv_12m']:,.0f}")
    col3.caption(f"{c['expected_purchases_12m']:.1f} expected purchase days")

    if c["repeat_purchase_days"] == 0:
        st.warning("This customer has bought on a single day. BG/NBD assumes such customers are still active, "
                   "and their CLV is essentially the average of all customers: treat it with caution.")
    else:
        st.metric("Probability of still being an active customer (BG/NBD)", f"{c['p_alive']:.0%}")

    st.markdown(f"**Recommended action ({c['segment']}):** {ACTIONS[c['segment']]}")

    st.divider()
    st.markdown("**Churn model, 10 Sep 2011 snapshot**")
    if pd.isna(c["churn_risk_sep_2011"]):
        st.info("Not in the September 2011 snapshot: no purchase in the 9 months before the cutoff.")
    else:
        outcome = "did not buy (churned)" if c["churned_after_sep_2011"] == 1 else "bought again"
        col1, col2 = st.columns(2)
        col1.metric("Predicted risk of not buying in Sep–Dec 2011", f"{c['churn_risk_sep_2011']:.0%}")
        col2.metric("What actually happened", outcome)

with campaign_tab:
    snapshot = customers.dropna(subset=["churn_risk_sep_2011"])
    st.markdown(
        "It is 10 September 2011 and marketing can contact a limited number of customers before Christmas. "
        "Which ones? Each rule below ranks the customers known at that date; since we know what happened "
        "afterwards, we can see how many of the customers who did **not** come back each rule would have reached."
    )

    size = st.slider("Customers to contact", min_value=50, max_value=1000, value=500, step=50)
    rule = st.selectbox("Targeting rule", list(campaign_lists(snapshot)))

    lists = campaign_lists(snapshot)
    results = pd.DataFrame({name: campaign_result(snapshot, order[:size]) for name, order in lists.items()}).T
    chosen = results.loc[rule]

    m1, m2, m3 = st.columns(3)
    m1.metric("Churners reached", f"{chosen['churners_reached']:,.0f}")
    m1.caption(f"out of {int(snapshot['churned_after_sep_2011'].sum()):,} churners in the snapshot")
    m2.metric("Share of contacted who churned", f"{chosen['churn_rate']:.0%}")
    m2.caption(f"{snapshot['churned_after_sep_2011'].mean():.0%} across all {len(snapshot):,} customers")
    m3.metric("Their spend in the previous 9 months", f"£{chosen['spend_of_churners']:,.0f}")

    chart_data = results.rename_axis("Rule").reset_index()
    colors = alt.Scale(domain=list(lists), range=[ORANGE, BLUE, AQUA, GREY])
    left, right = st.columns(2)
    for container, field, title, fmt in [
        (left, "churners_reached", "Churners reached", ",.0f"),
        (right, "spend_of_churners", "Spend of the churners reached (£)", ",.0f"),
    ]:
        chart = (alt.Chart(chart_data, title=title)
                 .mark_bar(cornerRadiusEnd=3)
                 .encode(
                     y=alt.Y("Rule:N", sort=list(lists), title=None, axis=alt.Axis(labelLimit=220)),
                     x=alt.X(f"{field}:Q", title=None),
                     color=alt.Color("Rule:N", scale=colors, legend=None),
                     opacity=alt.condition(alt.datum.Rule == rule, alt.value(1.0), alt.value(0.45)),
                     tooltip=[alt.Tooltip("Rule:N"), alt.Tooltip(f"{field}:Q", format=fmt)],
                 )
                 .properties(height=200))
        container.altair_chart(chart, width="stretch")

    st.caption(
        "Highest churn risk finds the most churners, which suits cheap automated campaigns. Highest revenue at "
        "risk (risk × past spend) finds fewer but more valuable ones, which suits personal win-back actions. "
        "Targeting by past spend alone mostly reaches loyal customers who would have bought anyway."
    )
