"""
Headless tests of the Streamlit dashboard with streamlit.testing.v1.AppTest (no browser needed).

AppTest runs the script like `streamlit run` would, exposes the rendered elements (metrics, widgets,
tabs, ...) and lets us interact with widgets and re-run the script.

Run from the repository root:
    pytest p04_visualisation_and_applications/c01_streamlit_app/test_main_app.py -q
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

PATH_APP = os.path.join(os.path.dirname(__file__), "main_app.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    """Run the app once (data generation and model training take a few seconds)."""
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


def test_prediction_form(app: AppTest):
    app.sidebar.multiselect(key="countries").set_value(["Denmark", "Germany"]).run()
    tickets = next(s for s in app.slider if s.label == "Support tickets")
    tickets.set_value(10)
    app.button[0].click().run()  # the form's submit button
    assert not app.exception
    probability = [m for m in app.metric if m.label == "Churn probability"]
    assert probability and probability[0].value.endswith("%")
