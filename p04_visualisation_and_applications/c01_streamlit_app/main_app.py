"""
Customer analytics dashboard with Streamlit and Plotly.

An interactive web app on top of the (synthetic) online-shop data of part 3:

- sidebar filters (date range, country, segment, plan) that apply to every chart on the page
- KPI tiles with the change versus the previous period of equal length
- Plotly charts: revenue over time and by category, churn rates and distributions
- a churn prediction panel: a scikit-learn pipeline trained on the fly, explained by its coefficients,
  and a form to score a hypothetical customer
- a data tab with the filtered transactions and a CSV download

Streamlit re-runs this script from top to bottom on every interaction. Expensive steps are therefore
cached: ``st.cache_data`` for data (a copy is returned on each call) and ``st.cache_resource`` for
the trained model (one shared object).

Run from the repository root:
    streamlit run p04_visualisation_and_applications/c01_streamlit_app/main_app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

# `streamlit run` puts the script's folder (not the repository root) on the path; add the root so that
# the absolute package imports below work without setting PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from p03_data_processing.c00_structured_data.utils.synthetic_data import (  # noqa: E402
    CATEGORIES,
    PLANS,
    SEGMENTS,
)
from p04_visualisation_and_applications.c00_plotting.utils.style import apply_style  # noqa: E402
from p04_visualisation_and_applications.c01_streamlit_app.utils import charts  # noqa: E402
from p04_visualisation_and_applications.c01_streamlit_app.utils.data import (  # noqa: E402
    Filters,
    apply_filters,
    churn_by,
    compute_kpis,
    load_data,
    monthly_revenue,
    previous_period,
    revenue_by,
)
from p04_visualisation_and_applications.c01_streamlit_app.utils.model import (  # noqa: E402
    TrainedModel,
    predict_churn,
    train_churn_model,
)

N_CUSTOMERS = 2000
SEED = 42


# 1. Cached loaders ---------------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating and cleaning data ...")
def get_data(n_customers: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the clean data once per (n_customers, seed); later calls hit the cache."""
    return load_data(n_customers=n_customers, seed=seed)


@st.cache_resource(show_spinner="Training churn model ...")
def get_model(n_customers: int, seed: int) -> TrainedModel:
    """Train the churn model once and share it between all sessions."""
    customers, transactions = get_data(n_customers, seed)
    return train_churn_model(customers, transactions, seed=seed)


# 2. Page sections ----------------------------------------------------------------------------------
def sidebar_filters(customers: pd.DataFrame, transactions: pd.DataFrame) -> Filters:
    """Render the filter widgets and return the selection."""
    st.sidebar.header("Filters")
    first, last = transactions["timestamp"].min().date(), transactions["timestamp"].max().date()
    default_start = (pd.Timestamp(last) - pd.DateOffset(months=12)).date()
    period = st.sidebar.date_input("Period", value=(default_start, last), min_value=first, max_value=last,
                                   key="period")
    # while the user is picking, the widget can return a single date
    start, end = (period if isinstance(period, (tuple, list)) and len(period) == 2 else (first, last))
    countries_all = list(customers["country"].cat.categories)
    countries = st.sidebar.multiselect("Country", countries_all, default=countries_all, key="countries")
    segments = st.sidebar.multiselect("Segment", SEGMENTS, default=SEGMENTS, key="segments")
    plans = st.sidebar.multiselect("Plan", PLANS, default=PLANS, key="plans")
    st.sidebar.caption("Data: synthetic online shop from part 3 (p03_data_processing).")
    return Filters(pd.Timestamp(start), pd.Timestamp(end), countries, segments, plans)


def kpi_row(transactions: pd.DataFrame, customers: pd.DataFrame, transactions_all: pd.DataFrame,
            customers_all: pd.DataFrame, filters: Filters) -> None:
    """KPI tiles with deltas versus the previous period."""
    kpis = compute_kpis(transactions, customers)
    _, prev_tx = apply_filters(customers_all, transactions_all, previous_period(filters))
    prev = compute_kpis(prev_tx, customers)

    def delta(key: str) -> str | None:
        return f"{(kpis[key] / prev[key] - 1):+.1%} vs. previous period" if prev[key] else None

    cols = st.columns(5)
    cols[0].metric("Net revenue", f"{kpis['revenue']:,.0f} EUR", delta("revenue"), border=True)
    cols[1].metric("Orders", f"{kpis['orders']:,}", delta("orders"), border=True)
    cols[2].metric("Active customers", f"{kpis['active_customers']:,}", delta("active_customers"), border=True)
    cols[3].metric("Avg. order value", f"{kpis['avg_order_value']:,.2f} EUR", delta("avg_order_value"),
                   border=True)
    cols[4].metric("Churn rate", f"{kpis['churn_rate']:.1%}", help="Share of the selected customers who churned",
                   border=True)


def overview_tab(transactions: pd.DataFrame) -> None:
    """Revenue over time and by category."""
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.revenue_over_time(monthly_revenue(transactions)), theme=None, key="revenue_time")
    by_category = revenue_by(transactions, "category")
    by_category["category"] = pd.Categorical(by_category["category"], categories=CATEGORIES)
    right.plotly_chart(charts.revenue_by_category(by_category), theme=None, key="revenue_category")


def customers_tab(customers: pd.DataFrame) -> None:
    """Churn rates per group and distributions of customer attributes."""
    overall = float(customers["churned"].mean())
    group = st.radio("Group churn rate by", ["segment", "plan", "country"], horizontal=True, key="group_by")
    left, right = st.columns(2)
    left.plotly_chart(charts.churn_rate_bars(churn_by(customers, group), group, overall), theme=None,
                      key="churn_bars")
    attribute = right.selectbox("Distribution of", ["support_tickets", "age", "annual_income"], key="attribute")
    right.plotly_chart(charts.histogram(customers, attribute, attribute.replace("_", " ")), theme=None,
                       key="histogram")


def model_tab(model: TrainedModel) -> None:
    """Model quality, explanation and an interactive prediction form."""
    st.markdown(f"Logistic regression on {len(model.feature_frame):,} training customers - "
                f"**test ROC AUC {model.test_auc:.3f}** (1.0 = perfect ranking, 0.5 = random).")
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.coefficient_chart(model.coefficients), theme=None, key="coefficients")

    X = model.feature_frame
    with right.form("prediction"):
        st.subheader("Score a customer")
        record = {
            "age": st.slider("Age", 18, 90, 40),
            "annual_income": st.number_input("Annual income (EUR)", 0, 500_000, int(X["annual_income"].median()),
                                             step=1000),
            "support_tickets": st.slider("Support tickets", 0, 15, 1),
            "n_transactions": st.slider("Purchases", 0, 40, int(X["n_transactions"].median())),
            "days_since_last_tx": st.slider("Days since last purchase", 0, 1000, 60),
            "tenure_days": st.slider("Days since signup", 30, 1100, 500),
            "segment": st.selectbox("Segment", SEGMENTS),
            "plan": st.selectbox("Plan", PLANS),
            "country": st.selectbox("Country", sorted(X["country"].dropna().unique())),
            "favourite_category": st.selectbox("Favourite category", CATEGORIES),
        }
        submitted = st.form_submit_button("Predict churn probability")
    # derived inputs: assume an average basket for the given number of purchases
    avg_amount = float(X["avg_amount"].median())
    record.update(avg_amount=avg_amount, total_spend=avg_amount * record["n_transactions"],
                  n_categories=min(record["n_transactions"], 3), refund_share=0.0)
    if submitted:
        probability = predict_churn(model, record)
        right.metric("Churn probability", f"{probability:.0%}")
        right.progress(probability)
        level = "high" if probability > 0.5 else "low"
        right.caption(f"Risk level: {level} (threshold 50 %).")


def data_tab(transactions: pd.DataFrame) -> None:
    """The filtered rows and a download button."""
    st.dataframe(transactions.head(1000), hide_index=True)
    st.caption(f"Showing the first {min(1000, len(transactions)):,} of {len(transactions):,} transactions.")
    st.download_button("Download filtered transactions (CSV)", transactions.to_csv(index=False).encode("utf-8"),
                       file_name="transactions_filtered.csv", mime="text/csv")


# 3. The app ----------------------------------------------------------------------------------------
def main() -> None:
    """Assemble the page."""
    st.set_page_config(page_title="Customer analytics", page_icon=":bar_chart:", layout="wide")
    apply_style()
    st.title("Customer analytics")
    st.caption("Interactive dashboard - all charts follow the filters in the sidebar.")

    customers_all, transactions_all = get_data(N_CUSTOMERS, SEED)
    filters = sidebar_filters(customers_all, transactions_all)
    customers, transactions = apply_filters(customers_all, transactions_all, filters)
    if transactions.empty or customers.empty:
        st.warning("No data for the current selection - widen the filters.")
        return

    kpi_row(transactions, customers, transactions_all, customers_all, filters)
    tab_overview, tab_customers, tab_model, tab_data = st.tabs(["Overview", "Customers", "Churn model", "Data"])
    with tab_overview:
        overview_tab(transactions)
    with tab_customers:
        customers_tab(customers)
    with tab_model:
        model_tab(get_model(N_CUSTOMERS, SEED))
    with tab_data:
        data_tab(transactions)


if __name__ == "__main__":
    main()
