## Interactive Applications with Streamlit
Charts in a report answer the questions their author anticipated. An interactive application lets the
reader ask their own: filter to a country, change the period, try "what if" inputs on a model.
Streamlit builds such applications from plain Python scripts, which makes it a popular choice for
dashboards, model demos and internal data tools.

### The Streamlit execution model
A Streamlit app is a script that is executed **from top to bottom on every interaction**. Widgets such as
`st.slider` or `st.multiselect` return their current value; when the user changes a widget, the script
is re-run with the new value and the page is re-rendered. This makes apps easy to write (no callbacks,
no HTML/JavaScript), but it means that expensive work must not be repeated on every rerun:

- `@st.cache_data` caches the *return value* of a function (data frames, query results) per combination
  of arguments; every caller receives a copy, so mutating it is safe.
- `@st.cache_resource` caches *one shared object* (a trained model, a database connection) for all users
  and sessions; it must not be mutated.
- `st.session_state` stores per-user state across reruns (not needed in this app).
- `st.form` groups widgets so that the script is only re-run when the form is submitted.

### The customer analytics dashboard (`main_app.py`)
The app visualises the cleaned online-shop data of part 3 (`load_clean_data()` in
`p03_data_processing/c00_structured_data/s01_preprocessing/utils/cleaning.py`):

- **Sidebar filters**: period (`st.date_input` with a range; by default the last 12 calendar months, so
  that the monthly chart does not start with a partial month), country, segment and plan
  (`st.multiselect`). The filters apply to the KPIs, the charts and the data table (not to the churn
  model, see below); an empty selection shows a warning instead of empty charts. Date ranges are
  half-open - from the start date up to, but excluding, the day after the end date - so consecutive
  periods never count a transaction twice.
- **KPI tiles** (`st.metric`): net revenue, orders, active customers, average order value and churn rate,
  with the change versus the previous period of equal length. If the data does not cover that previous
  period completely, no change is shown: comparing with a half-empty period would produce meaningless
  growth rates of several hundred percent. A headline number is better shown as a number than as a
  chart.
- **Tabs** (`st.tabs`):
  - *Overview*: monthly net revenue (line) and revenue by category (sorted bars),
  - *Customers*: churn rate by segment/plan/country with the overall rate as a reference line, and the
    distribution of an attribute for churned vs. active customers,
  - *Churn model*: the logistic regression pipeline of part 3 trained on the fly on all customers
    (cached with `st.cache_resource`), its test ROC AUC (about 0.77), the strongest associations
    (largest coefficients), and a form to score a hypothetical customer. Like in part 3, the model
    deliberately does not use the days since the last purchase: churned customers stop buying, so that
    feature is a consequence of churn and would leak the outcome (a test ROC AUC of 0.94 that is
    useless in practice). For the same reason the chart is titled "associations", not "drivers": a
    coefficient describes a correlation in the data, not a cause,
  - *Data*: the filtered transactions (`st.dataframe`) and a CSV download (`st.download_button`).

The code is split so that the Streamlit script stays thin: `utils/data.py` (loading, filtering,
aggregations, KPIs), `utils/model.py` (training and prediction) and `utils/charts.py` (Plotly figures with
the shared template of chapter `c00_plotting`) contain no Streamlit code and can be tested and reused
independently. Charts are rendered with `st.plotly_chart(fig, theme=None)`, which keeps our own Plotly
template instead of Streamlit's theme.

`streamlit run` adds the folder of the script - not the repository root - to Python's import path. The
app therefore inserts the repository root into `sys.path` before importing the project packages, so it
also works without setting `PYTHONPATH`.

### Testing the app
`streamlit.testing.v1.AppTest` runs an app headlessly, exposes the rendered elements and lets tests
interact with widgets:

```python
at = AppTest.from_file("main_app.py", default_timeout=120)
at.run()
assert not at.exception
at.sidebar.multiselect(key="countries").set_value(["Denmark"]).run()
```

`test_main_app.py` contains unit tests of the plain functions in `utils/` (half-open date filters, the
previous period, the KPIs, and a churn model without post-outcome features whose prediction rises with
the number of support tickets) and headless app tests: the app runs without exceptions, the KPIs and
tabs are rendered, the filters change the KPIs (an empty selection shows a warning), no change is shown
when the previous period is not covered by the data, and the prediction form returns a probability
between 0 and 1 that reacts to its inputs. Each app test starts from a fresh `AppTest`, so the tests do
not depend on each other's widget state; data and model come from Streamlit's cache.

### Deployment
For a quick demo, `streamlit run` on a laptop is enough. To share an app, common options are Streamlit
Community Cloud (connects to a GitHub repository), a Docker container on any container platform
(the app listens on port 8501 by default; `/_stcore/health` serves as a health check), or Hugging Face
Spaces. Containerisation and cloud deployment are covered in part 1.
