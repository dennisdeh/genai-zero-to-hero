"""
Plotly figures for the dashboard, styled with the shared template of chapter c00_plotting.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from p04_visualisation_and_applications.c00_plotting.utils.style import CATEGORICAL, GREY


def revenue_over_time(monthly: pd.DataFrame) -> go.Figure:
    """Monthly net revenue as a single line (one series: no legend needed, the title names it)."""
    fig = px.line(monthly, x="month", y="revenue", markers=True, color_discrete_sequence=[CATEGORICAL[0]],
                  labels={"month": "", "revenue": "Net revenue (EUR)"}, title="Monthly net revenue")
    fig.update_traces(hovertemplate="%{x|%b %Y}: %{y:,.0f} EUR<extra></extra>")
    fig.update_layout(hovermode="x", height=360, margin=dict(t=50))
    return fig


def revenue_by_category(by_category: pd.DataFrame) -> go.Figure:
    """Sorted horizontal bars in a single colour (categories have no order -> no colour ramp)."""
    df = by_category.sort_values("revenue")
    fig = px.bar(df, x="revenue", y="category", orientation="h", text_auto=",.0f",
                 color_discrete_sequence=[CATEGORICAL[0]],
                 labels={"revenue": "Net revenue (EUR)", "category": ""}, title="Revenue by category")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=360, margin=dict(t=50))
    return fig


def churn_rate_bars(churn: pd.DataFrame, column: str, overall: float) -> go.Figure:
    """Churn rate per group with the overall rate as a reference line."""
    fig = px.bar(churn, x=column, y="churn_rate", text_auto=".0%", color_discrete_sequence=[CATEGORICAL[0]],
                 hover_data={"customers": True}, labels={"churn_rate": "Churn rate", column: ""},
                 title=f"Churn rate by {column}")
    fig.add_hline(y=overall, line=dict(color=GREY, width=1), annotation_text=f"overall {overall:.0%}",
                  annotation_position="top left")
    fig.update_layout(yaxis_tickformat=".0%", height=360, margin=dict(t=50))
    return fig


def histogram(customers: pd.DataFrame, column: str, label: str) -> go.Figure:
    """Distribution of a numeric customer attribute split by churn (two overlaid series, legend shown)."""
    df = customers.assign(status=customers["churned"].map({True: "churned", False: "active"}))
    fig = px.histogram(df, x=column, color="status", barmode="overlay", opacity=0.7, nbins=30,
                       category_orders={"status": ["active", "churned"]},
                       color_discrete_sequence=CATEGORICAL[:2], labels={column: label, "status": ""},
                       title=f"{label}: churned vs. active customers")
    fig.update_layout(yaxis_title="Customers", height=360, margin=dict(t=50), bargap=0.05)
    return fig


def coefficient_chart(coefficients: pd.Series, n: int = 10) -> go.Figure:
    """Largest logistic regression coefficients; sign shown by colour AND by direction of the bar."""
    top = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index).head(n)[::-1]
    names = [name.split("__", 1)[-1] for name in top.index]
    colours = [CATEGORICAL[7] if v > 0 else CATEGORICAL[0] for v in top]
    fig = go.Figure(go.Bar(x=top.values, y=names, orientation="h", marker_color=colours,
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_layout(title="What drives churn? (standardised coefficients, red = increases risk)",
                      xaxis_title="Coefficient", height=380, margin=dict(t=50))
    return fig

