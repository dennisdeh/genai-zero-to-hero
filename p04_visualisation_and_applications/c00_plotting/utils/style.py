"""
A consistent visual style for all Plotly figures of this part, plus helpers to save figures.

Good charts share one visual system: the same fonts, recessive grid lines, thin marks and a fixed
categorical colour order (a series keeps its colour in every chart). The palette below is a
colour-vision-deficiency (CVD) checked set of 8 categorical hues; the order matters - neighbouring
slots stay distinguishable for colour-blind readers. Use at most ~3 colours when marks overlap
(scatter plots), otherwise highlight one series and grey out the rest, or use small multiples.

Usage:
    from p04_visualisation_and_applications.c00_plotting.utils.style import apply_style, save_html
    apply_style()  # registers the "gzh" template and makes it the default
"""

import os

import plotly.graph_objects as go
import plotly.io as pio

# categorical slots in fixed order (light mode)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# sequential: one hue (blue), light -> dark
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# diverging: blue <-> neutral grey <-> red (for signed values such as correlations)
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec8a89"], [1.0, "#a8302f"]]
GREY = "#c3c2b7"  # de-emphasised marks
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def build_template() -> go.layout.Template:
    """Plotly template with recessive axes, hairline grid, readable fonts and the categorical palette."""
    axis = dict(gridcolor=GRID, gridwidth=1, linecolor=GRID, zerolinecolor=GRID, ticks="",
                tickfont=dict(color=INK_MUTED), title=dict(font=dict(color=INK_SECONDARY)))
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT, size=13, color=INK),
            title=dict(font=dict(size=18, color=INK), x=0.0, xanchor="left"),
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            colorway=CATEGORICAL,
            colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
                            diverging=DIVERGING),
            xaxis=axis,
            yaxis=axis,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_SECONDARY)),
            hoverlabel=dict(font=dict(family=FONT)),
            margin=dict(l=60, r=30, t=70, b=50),
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2), marker=dict(size=8, line=dict(width=1, color=SURFACE)))],
            bar=[go.Bar(marker=dict(line=dict(width=0)))],
        ),
    )


def apply_style() -> None:
    """Register the template as 'gzh' and use it on top of 'plotly_white' for all new figures."""
    pio.templates["gzh"] = build_template()
    pio.templates.default = "plotly_white+gzh"


def save_html(fig: go.Figure, name: str, path_outputs: str, offline: bool = True) -> str:
    """
    Save a figure as a standalone, interactive HTML file.

    :param fig: the figure
    :param name: file name without extension
    :param path_outputs: output folder (created if needed)
    :param offline: embed plotly.js (~4 MB, works without internet) or load it from a CDN (small file)
    :return: path of the written file
    """
    os.makedirs(path_outputs, exist_ok=True)
    path = os.path.join(path_outputs, f"{name}.html")
    fig.write_html(path, include_plotlyjs=True if offline else "cdn", full_html=True)
    return path


def save_report(figures: dict[str, go.Figure], path: str, title: str = "Report") -> str:
    """
    Combine several figures into one HTML page (plotly.js is embedded only once).

    :param figures: {section title: figure}
    :param path: output file path
    :param title: page title
    :return: path of the written file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title></head>",
             f"<body style='font-family:{FONT};background:{SURFACE};max-width:1100px;margin:auto'>",
             f"<h1>{title}</h1>"]
    for i, (heading, fig) in enumerate(figures.items()):
        parts.append(f"<h2>{heading}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path
