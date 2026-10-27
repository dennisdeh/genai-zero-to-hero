"""
Data access and aggregation for the customer analytics dashboard (no Streamlit code here, so the
functions are easy to test and reuse).

The data is the synthetic online-shop dataset of part 3 (p03_data_processing), generated and cleaned
on the fly with the cleaning procedure of the preprocessing section.
"""

from dataclasses import dataclass

import pandas as pd

from p03_data_processing.c00_structured_data.s01_preprocessing.utils.cleaning import load_clean_data


@dataclass
class Filters:
    """The current selection in the sidebar."""

    start: pd.Timestamp
    end: pd.Timestamp
    countries: list[str]
    segments: list[str]
    plans: list[str]


def load_data(n_customers: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean customers and transactions; transactions get the customer attributes joined for filtering."""
    customers, transactions = load_clean_data(n_customers=n_customers, seed=seed)
    customers = customers.assign(country=customers["country"].cat.add_categories("Unknown").fillna("Unknown"))
    transactions = transactions.merge(customers[["customer_id", "country", "segment", "plan"]], on="customer_id")
    return customers, transactions


def apply_filters(customers: pd.DataFrame, transactions: pd.DataFrame,
                  filters: Filters) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict customers by attributes and transactions by attributes and date range.

    The date range is half-open, [start, end + 1 day): it includes the whole end day, and consecutive
    periods do not overlap (a transaction at midnight is not counted twice).
    """
    mask_c = (customers["country"].isin(filters.countries) & customers["segment"].isin(filters.segments)
              & customers["plan"].isin(filters.plans))
    mask_t = (transactions["country"].isin(filters.countries) & transactions["segment"].isin(filters.segments)
              & transactions["plan"].isin(filters.plans)
              & (transactions["timestamp"] >= filters.start)
              & (transactions["timestamp"] < filters.end + pd.Timedelta(days=1)))
    return customers[mask_c], transactions[mask_t]


def compute_kpis(transactions: pd.DataFrame, customers: pd.DataFrame) -> dict[str, float]:
    """Headline numbers: revenue (net of refunds), orders, active customers, average order value, churn rate."""
    purchases = transactions[~transactions["is_refund"]]
    n_orders = len(purchases)
    return {
        "revenue": float(transactions["amount"].sum()),
        "orders": n_orders,
        "active_customers": int(purchases["customer_id"].nunique()),
        "avg_order_value": float(purchases["amount"].mean()) if n_orders else 0.0,
        "churn_rate": float(customers["churned"].mean()) if len(customers) else 0.0,
    }


def previous_period(filters: Filters, first_date: pd.Timestamp | None = None) -> Filters | None:
    """
    The period of equal length directly before the selected one (for KPI deltas).

    :param first_date: first date with data; if the previous period starts earlier, it is incomplete
        and a comparison would be misleading, so None is returned
    """
    length = filters.end - filters.start
    previous = Filters(filters.start - length - pd.Timedelta(days=1), filters.start - pd.Timedelta(days=1),
                       filters.countries, filters.segments, filters.plans)
    if first_date is not None and previous.start < pd.Timestamp(first_date).normalize():
        return None
    return previous


def monthly_revenue(transactions: pd.DataFrame) -> pd.DataFrame:
    """Net revenue and number of orders per month."""
    df = transactions.assign(month=transactions["timestamp"].dt.to_period("M").dt.to_timestamp())
    return df.groupby("month").agg(revenue=("amount", "sum"), orders=("transaction_id", "count")).reset_index()


def revenue_by(transactions: pd.DataFrame, column: str) -> pd.DataFrame:
    """Net revenue per value of a categorical column, sorted descending."""
    return (transactions.groupby(column, observed=True)["amount"].sum().rename("revenue")
            .sort_values(ascending=False).reset_index())


def churn_by(customers: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate and number of customers per value of a categorical column."""
    return (customers.groupby(column, observed=True)
            .agg(churn_rate=("churned", "mean"), customers=("customer_id", "count")).reset_index())
