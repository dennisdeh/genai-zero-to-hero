# Plotting with Plotly

## Environment
- Python conda environment to use: `p04_visualisation_and_applications/conda_env_gzh_viz0.yml`
  (`conda env create -f p04_visualisation_and_applications/conda_env_gzh_viz0.yml && conda activate gzh_viz0`)
- No docker setup or environment variables are needed. The datasets ship with Plotly (`plotly.express.data`).
- Optional: static image export (`fig.write_image`) needs the `kaleido` package and a Chrome/Chromium browser;
  without it the scripts skip the PNG export.

## How to run
From the repository root:

```bash
# plotly.express: distributions, highlighting, small multiples, ranking, scatter matrix, animation
python -m p04_visualisation_and_applications.c00_plotting.main_1_plotly_express
# plotly.graph_objects: time series with range slider, correlation heatmap, subplots, export
python -m p04_visualisation_and_applications.c00_plotting.main_2_graph_objects
```

The figures are written to `c00_plotting/outputs/`. Open `report_plotly_express.html` and
`report_graph_objects.html` in a browser - they contain all figures and work offline. The individual
`px_*.html`/`go_*.html` files load plotly.js from a CDN.

## Structure
- `utils/style.py`: shared Plotly template (palette, fonts, grid) and helpers to save figures and reports
