# Customer analytics dashboard with Streamlit

## Environment
- Python conda environment to use: `p04_visualisation_and_applications/conda_env_gzh_viz0.yml`
  (`conda env create -f p04_visualisation_and_applications/conda_env_gzh_viz0.yml && conda activate gzh_viz0`)
- No docker setup, environment variables or internet access are needed. The app uses the synthetic data and
  the cleaning/feature code of `p03_data_processing/c00_structured_data` (imported from the repository).

## How to run
From the repository root:

```bash
streamlit run p04_visualisation_and_applications/c01_streamlit_app/main_app.py
```

Streamlit prints the local URL (default http://localhost:8501) and opens it in the browser. Useful options:
`--server.port 8599` (another port), `--server.headless true` (do not open a browser, e.g. on a server).
The health endpoint `http://localhost:8501/_stcore/health` returns `ok` when the server is up.

Run the headless tests of the app (Streamlit's `AppTest`, no browser needed):

```bash
pytest p04_visualisation_and_applications/c01_streamlit_app/test_main_app.py -q
```

## Structure
- `main_app.py`: the Streamlit app (layout, widgets, caching)
- `utils/data.py`: loading, filtering, KPIs and aggregations (no Streamlit code)
- `utils/model.py`: training and using the churn model (re-uses the p03 feature pipeline)
- `utils/charts.py`: Plotly figures in the shared style of `c00_plotting/utils/style.py`
- `test_main_app.py`: AppTest-based tests
