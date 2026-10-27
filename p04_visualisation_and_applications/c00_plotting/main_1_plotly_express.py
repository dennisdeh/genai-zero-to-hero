"""
Interactive charts with plotly.express - one function call per chart.

Using datasets that ship with plotly (no download needed), the script shows the main chart families
and the design choices behind them:

1. distributions: histogram with a marginal box plot, and box plots per group
2. relationships: a scatter plot that highlights one group and greys out the rest
3. small multiples (faceting) instead of one overloaded chart
4. ranking: a sorted horizontal bar chart
5. many variables at once: a scatter matrix
6. change over time: an animated bubble chart

Every figure is saved as an interactive HTML file in outputs/, and all of them are combined into
one report page (outputs/report_plotly_express.html).

Run from the repository root:
    python -m p04_visualisation_and_applications.c00_plotting.main_1_plotly_express
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from p04_visualisation_and_applications.c00_plotting.utils.style import (
    CATEGORICAL,
    GREY,
    apply_style,
    save_html,
    save_report,
)

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")


def distribution_histogram(tips: pd.DataFrame) -> go.Figure:
    """Distribution of the bill amount for lunch vs. dinner, with marginal box plots."""
    fig = px.histogram(tips, x="total_bill", color="time", marginal="box", nbins=30, barmode="overlay",
                       opacity=0.75, labels={"total_bill": "Total bill (USD)", "time": "Meal"},
                       title="Dinner bills are higher and more spread out than lunch bills")
    fig.update_layout(yaxis_title="Number of bills", bargap=0.05)
    return fig


def distribution_box(tips: pd.DataFrame) -> go.Figure:
    """Tip percentage per weekday as box plots with all points (small data: show the raw points)."""
    tips = tips.assign(tip_pct=100 * tips["tip"] / tips["total_bill"])
    fig = px.box(tips, x="day", y="tip_pct", points="all", category_orders={"day": ["Thur", "Fri", "Sat", "Sun"]},
                 labels={"tip_pct": "Tip (% of bill)", "day": "Day"}, color_discrete_sequence=[CATEGORICAL[0]],
                 title="Tip percentages are similar across days - a few generous outliers")
    return fig


def highlight_scatter(gapminder: pd.DataFrame, highlight: str = "Africa") -> go.Figure:
    """GDP per capita vs. life expectancy in 2007; one continent highlighted, the rest in grey."""
    df = gapminder[gapminder["year"] == 2007].copy()
    df["group"] = df["continent"].where(df["continent"] == highlight, "Other continents")
    fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", size_max=45, color="group", log_x=True,
                     hover_name="country", hover_data={"group": False, "pop": ":,", "gdpPercap": ":,.0f"},
                     color_discrete_map={highlight: CATEGORICAL[1], "Other continents": GREY},
                     category_orders={"group": ["Other continents", highlight]},
                     labels={"gdpPercap": "GDP per capita (USD, log scale)", "lifeExp": "Life expectancy (years)",
                             "group": ""},
                     title=f"Richer countries live longer - {highlight} highlighted (2007)")
    return fig


def small_multiples(gapminder: pd.DataFrame) -> go.Figure:
    """Life expectancy over time, one panel per continent, one thin line per country."""
    fig = px.line(gapminder, x="year", y="lifeExp", line_group="country", facet_col="continent",
                  facet_col_wrap=3, hover_name="country", color_discrete_sequence=[CATEGORICAL[0]],
                  labels={"lifeExp": "Life expectancy", "year": ""},
                  title="Life expectancy rose almost everywhere (small multiples per continent)")
    fig.update_traces(line=dict(width=1), opacity=0.5)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # "continent=Asia" -> "Asia"
    return fig


def ranking_bar(gapminder: pd.DataFrame, n: int = 15) -> go.Figure:
    """Top-n countries by GDP per capita in 2007 as a sorted horizontal bar chart (one colour)."""
    df = gapminder[gapminder["year"] == 2007].nlargest(n, "gdpPercap").sort_values("gdpPercap")
    fig = px.bar(df, x="gdpPercap", y="country", orientation="h", text_auto=",.0f",
                 color_discrete_sequence=[CATEGORICAL[0]],
                 labels={"gdpPercap": "GDP per capita (USD)", "country": ""},
                 title=f"Top {n} countries by GDP per capita (2007)")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def scatter_matrix(iris: pd.DataFrame) -> go.Figure:
    """Pairwise relationships of the four iris measurements (three species = three colours)."""
    dims = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    fig = px.scatter_matrix(iris, dimensions=dims, color="species", opacity=0.7,
                            labels={d: d.replace("_", " ") for d in dims},
                            title="Petal measurements separate the iris species")
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=6))
    fig.update_layout(height=700)
    return fig


def animated_bubbles(gapminder: pd.DataFrame) -> go.Figure:
    """
    Gapminder-style animation over the years (press play).

    The axis ranges are fixed for all frames (otherwise the axes would jump) and wide enough for every
    country and year, including Kuwait's very high GDP in the 1950s-70s and Rwanda's life expectancy
    of 24 years in 1992.
    """
    fig = px.scatter(gapminder, x="gdpPercap", y="lifeExp", size="pop", size_max=45, animation_frame="year",
                     animation_group="country", hover_name="country", log_x=True, range_x=[200, 150_000],
                     range_y=[20, 90], color_discrete_sequence=[CATEGORICAL[0]],
                     labels={"gdpPercap": "GDP per capita (USD, log scale)", "lifeExp": "Life expectancy"},
                     title="Development of health and wealth 1952-2007 (press play)")
    return fig


if __name__ == "__main__":
    # 1. Style and data (all bundled with plotly)
    apply_style()
    tips, gapminder, iris = px.data.tips(), px.data.gapminder(), px.data.iris()
    print(f"tips: {tips.shape}, gapminder: {gapminder.shape}, iris: {iris.shape}")

    # 2. Build the figures
    figures = {
        "Distribution: histogram with marginal box plot": distribution_histogram(tips),
        "Distribution per group: box plots with points": distribution_box(tips),
        "Relationship: highlight instead of eight colours": highlight_scatter(gapminder),
        "Small multiples": small_multiples(gapminder),
        "Ranking: sorted bar chart": ranking_bar(gapminder),
        "Many variables: scatter matrix": scatter_matrix(iris),
        "Change over time: animation": animated_bubbles(gapminder),
    }

    # 3. Save individual HTML files (small files that load plotly.js from a CDN, i.e. need internet to view)
    for i, (title, fig) in enumerate(figures.items()):
        name = f"px_{i + 1}_" + title.split(":")[0].lower().replace(" ", "_")
        path = save_html(fig, name, PATH_OUTPUTS, offline=False)
        print(f"saved {path}")

    # 4. One report page with all figures (plotly.js embedded once, works offline)
    path = save_report(figures, os.path.join(PATH_OUTPUTS, "report_plotly_express.html"),
                       title="Plotly Express examples")
    print(f"report: {path} - open it in a browser")
