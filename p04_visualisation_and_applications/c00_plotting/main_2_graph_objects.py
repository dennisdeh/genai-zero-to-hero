"""
Full control with plotly.graph_objects: time series, heatmaps, subplots and annotations.

plotly.express covers most needs with one call; plotly.graph_objects (go) builds a figure trace by
trace and gives access to every layout detail. Using the weekly stock prices bundled with plotly
(2018-2019, normalised to 1 at the start) the script shows

1. a multi-series time series with a range slider and range selector buttons
2. a correlation heatmap of weekly returns with a diverging colour scale centred at zero
3. stacked subplots sharing the x-axis (price and drawdown) instead of a misleading dual y-axis,
   with a shaded region and an annotation marking the maximum drawdown
4. exporting: standalone HTML, JSON (to rebuild the figure elsewhere) and - if kaleido is
   installed - a static PNG

Run from the repository root:
    python -m p04_visualisation_and_applications.c00_plotting.main_2_graph_objects
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from p04_visualisation_and_applications.c00_plotting.utils.style import (
    CATEGORICAL,
    DIVERGING,
    INK_SECONDARY,
    apply_style,
    save_html,
    save_report,
)

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
TICKERS = ["AAPL", "AMZN", "FB", "GOOG", "MSFT", "NFLX"]  # fixed order -> fixed colour per ticker


def load_prices() -> pd.DataFrame:
    """Weekly normalised prices indexed by date."""
    prices = px.data.stocks()
    prices["date"] = pd.to_datetime(prices["date"])
    return prices.set_index("date")[TICKERS]


def time_series_with_slider(prices: pd.DataFrame) -> go.Figure:
    """One line per ticker (indexed to 100), with range slider, range buttons and a unified hover."""
    fig = go.Figure()
    for ticker, colour in zip(TICKERS, CATEGORICAL):
        fig.add_trace(go.Scatter(x=prices.index, y=100 * prices[ticker], name=ticker, mode="lines",
                                 line=dict(color=colour),
                                 hovertemplate="%{y:.1f}"))
        # direct label at the end of each line (identity is not carried by colour alone)
        fig.add_annotation(x=prices.index[-1], y=100 * prices[ticker].iloc[-1], text=ticker, showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=INK_SECONDARY, size=11))
    fig.add_hline(y=100, line=dict(color="#c3c2b7", width=1))
    fig.update_layout(
        title="Stock prices indexed to 100 at the start of 2018",
        yaxis_title="Index (2018-01-01 = 100)",
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.08),
            rangeselector=dict(buttons=[
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="All"),
            ]),
        ),
        height=550,
    )
    return fig


def correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    """Correlation of weekly returns; diverging scale fixed to [-1, 1] so that white means zero."""
    corr = prices.pct_change().dropna().corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=corr.columns, y=corr.index, zmin=-1, zmax=1, zmid=0,
        colorscale=DIVERGING,  # blue (negative) - grey (zero) - red (positive), see style.py
        texttemplate="%{z:.2f}", hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="corr."),
    ))
    fig.update_layout(title="Correlation of weekly returns", yaxis=dict(autorange="reversed"), height=500,
                      width=600)
    return fig


def price_and_drawdown(prices: pd.DataFrame, ticker: str = "NFLX") -> go.Figure:
    """Two stacked panels sharing the x-axis: price and drawdown from the running maximum."""
    price = 100 * prices[ticker]
    drawdown = 100 * (price / price.cummax() - 1)
    trough = drawdown.idxmin()
    peak = price.loc[:trough].idxmax()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06,
                        subplot_titles=(f"{ticker} price (indexed)", "Drawdown from previous high (%)"))
    fig.add_trace(go.Scatter(x=price.index, y=price, name="price", line=dict(color=CATEGORICAL[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="drawdown", fill="tozeroy",
                             line=dict(color=CATEGORICAL[7], width=1)), row=2, col=1)
    fig.add_vrect(x0=peak, x1=trough, fillcolor="#e1e0d9", opacity=0.5, line_width=0, layer="below")
    fig.add_annotation(x=trough, y=drawdown.min(), text=f"max drawdown {drawdown.min():.0f}%", showarrow=True,
                       arrowhead=0, ax=40, ay=20, row=2, col=1)
    fig.update_layout(title=f"{ticker}: price and drawdown - two panels instead of two y-axes",
                      showlegend=False, height=600, hovermode="x unified")
    return fig


def export(fig: go.Figure, name: str) -> None:
    """Save HTML and JSON; PNG only if the optional kaleido package (and a Chrome browser) is available."""
    print(f"saved {save_html(fig, name, PATH_OUTPUTS, offline=False)}")
    path_json = os.path.join(PATH_OUTPUTS, f"{name}.json")
    with open(path_json, "w", encoding="utf-8") as f:
        f.write(pio.to_json(fig))
    with open(path_json, encoding="utf-8") as f:
        restored = pio.from_json(f.read())
    print(f"saved {path_json} (restored figure has {len(restored.data)} traces)")
    try:
        fig.write_image(os.path.join(PATH_OUTPUTS, f"{name}.png"), scale=2)
        print(f"saved {name}.png")
    except Exception as e:  # kaleido missing or no browser available
        print(f"(static PNG export skipped: {type(e).__name__})")


if __name__ == "__main__":
    # 1. Style and data
    apply_style()
    prices = load_prices()
    print(f"prices: {prices.shape[0]} weeks from {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")

    # 2. Figures
    figures = {
        "Time series with range slider": time_series_with_slider(prices),
        "Correlation heatmap": correlation_heatmap(prices),
        "Subplots: price and drawdown": price_and_drawdown(prices),
    }

    # 3. Export
    for name, fig in zip(["go_1_time_series", "go_2_heatmap", "go_3_subplots"], figures.values()):
        export(fig, name)
    path = save_report(figures, os.path.join(PATH_OUTPUTS, "report_graph_objects.html"),
                       title="Plotly graph_objects examples")
    print(f"report: {path} - open it in a browser")
