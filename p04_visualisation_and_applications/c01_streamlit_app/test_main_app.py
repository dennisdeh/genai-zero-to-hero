"""
Tests of the Streamlit dashboard.

- Unit tests of the plain functions in utils/ (filters, previous period, KPIs, churn model).
- Headless tests of the app with streamlit.testing.v1.AppTest (no browser needed): AppTest runs the
  script like `streamlit run` would, exposes the rendered elements (metrics, widgets, tabs, ...) and
  lets us interact with widgets and re-run the script.

Run from the repository root:
    pytest p04_visualisation_and_applications/c01_streamlit_app/test_main_app.py -q
"""

import os
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from p04_visualisation_and_applications.c01_streamlit_app.utils.data import (  # noqa: E402
    Filters,
    apply_filters,
    compute_kpis,
    previous_period,
)
from p04_visualisation_and_applications.c01_streamlit_app.utils.model import (  # noqa: E402
    predict_churn,
    train_churn_model,
)

PATH_APP = os.path.join(os.path.dirname(__file__), "main_app.py")


# ----------------------------------------------------------------------------------------------------
# unit tests of utils/ (small hand-made data, fast)
# ----------------------------------------------------------------------------------------------------
@pytest.fixture()
def small_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.DataFrame({"customer_id": ["A", "B"], "country": ["Denmark", "Sweden"],
                              "segment": ["Consumer", "Corporate"], "plan": ["Basic", "Premium"],
                              "churned": [True, False]})
    transactions = pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4"],
        "customer_id": ["A", "A", "B", "B"],
        "timestamp": pd.to_datetime(["2024-01-31 23:00", "2024-02-01 00:00", "2024-02-15 12:00",
                                     "2024-02-29 23:59"]),
        "amount": [100.0, 50.0, -20.0, 30.0],
        "is_refund": [False, False, True, False],
    }).merge(customers[["customer_id", "country", "segment", "plan"]], on="customer_id")
    return customers, transactions


def _filters(start: str, end: str) -> Filters:
    return Filters(pd.Timestamp(start), pd.Timestamp(end), ["Denmark", "Sweden"], ["Consumer", "Corporate"],
                   ["Basic", "Premium"])


def test_apply_filters_half_open_dates(small_data):
    customers, transactions = small_data
    _, jan = apply_filters(customers, transactions, _filters("2024-01-01", "2024-01-31"))
    _, feb = apply_filters(customers, transactions, _filters("2024-02-01", "2024-02-29"))
    assert list(jan["transaction_id"]) == ["t1"]  # the whole end day is included ...
    assert list(feb["transaction_id"]) == ["t2", "t3", "t4"]  # ... and midnight belongs to the next day only
    country_filter = Filters(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-29"), ["Sweden"],
                             ["Consumer", "Corporate"], ["Basic", "Premium"])
    c, t = apply_filters(customers, transactions, country_filter)
    assert list(c["customer_id"]) == ["B"] and set(t["customer_id"]) == {"B"}


def test_previous_period():
    current = _filters("2024-02-01", "2024-02-29")  # 29 days
    previous = previous_period(current)
    assert previous.end == pd.Timestamp("2024-01-31")
    assert previous.end - previous.start == current.end - current.start  # equal length, no overlap
    assert previous_period(current, first_date=pd.Timestamp("2024-01-03 10:00")) == previous
    assert previous_period(current, first_date=pd.Timestamp("2024-01-05")) is None  # not covered by the data


def test_compute_kpis(small_data):
    customers, transactions = small_data
    kpis = compute_kpis(transactions, customers)
    assert kpis["revenue"] == pytest.approx(160.0)  # net of the refund
    assert kpis["orders"] == 3 and kpis["active_customers"] == 2
    assert kpis["avg_order_value"] == pytest.approx(60.0)
    assert kpis["churn_rate"] == pytest.approx(0.5)
    assert compute_kpis(transactions.iloc[:0], customers.iloc[:0])["avg_order_value"] == 0.0


def test_churn_model_has_no_post_outcome_features():
    from p03_data_processing.c00_structured_data.s01_preprocessing.utils.cleaning import load_clean_data

    model = train_churn_model(*load_clean_data(n_customers=600, seed=1), seed=1)
    assert "days_since_last_tx" not in model.feature_frame.columns
    assert 0.5 < model.test_auc < 0.9  # an honest model is useful, but far from perfect
    record = model.feature_frame.median(numeric_only=True).to_dict() | {
        "country": "Denmark", "segment": "Consumer", "favourite_category": "Groceries", "plan": "Basic"}
    low = predict_churn(model, record | {"support_tickets": 0})
    high = predict_churn(model, record | {"support_tickets": 10})
    assert 0.0 <= low < high <= 1.0  # more support tickets -> higher churn risk


# ----------------------------------------------------------------------------------------------------
# headless app tests (each test gets a fresh app; data and model come from Streamlit's cache)
# ----------------------------------------------------------------------------------------------------
@pytest.fixture()
def app() -> AppTest:
    """Run the app (the first run generates the data and trains the model, later runs hit the cache)."""
    at = AppTest.from_file(PATH_APP, default_timeout=120)
    at.run()
    return at


def test_app_runs_without_exception(app: AppTest):
    assert not app.exception
    assert app.title[0].value == "Customer analytics"


def test_kpis_and_tabs_rendered(app: AppTest):
    labels = [m.label for m in app.metric]
    assert labels[:5] == ["Net revenue", "Orders", "Active customers", "Avg. order value", "Churn rate"]
    assert [t.label for t in app.tabs] == ["Overview", "Customers", "Churn model", "Data"]


def test_filter_changes_kpis(app: AppTest):
    orders_before = app.metric[1].value
    app.sidebar.multiselect(key="countries").set_value(["Denmark"]).run()
    assert not app.exception
    assert app.metric[1].value != orders_before
    app.sidebar.multiselect(key="countries").set_value([]).run()  # empty selection -> friendly warning
    assert not app.exception
    assert len(app.warning) == 1


def test_no_delta_without_complete_previous_period(app: AppTest):
    first = app.sidebar.date_input(key="period").min
    app.sidebar.date_input(key="period").set_value((first, first + pd.Timedelta(days=90))).run()
    assert not app.exception
    assert not any(m.delta for m in app.metric[:5])  # no delta is rendered


def _predict(app: AppTest, tickets: int) -> float:
    next(s for s in app.slider if s.label == "Support tickets").set_value(tickets)
    app.button[0].click().run()  # the form's submit button
    assert not app.exception
    probability = [m for m in app.metric if m.label == "Churn probability"]
    assert probability and probability[0].value.endswith("%")
    return float(probability[0].value.rstrip("%")) / 100


def test_prediction_form(app: AppTest):
    low, high = _predict(app, tickets=0), _predict(app, tickets=10)
    assert 0.0 <= low < high <= 1.0
