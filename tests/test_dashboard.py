from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

from benchmark_analysis import run_benchmarks  # noqa: E402
from dashboard import data as store  # noqa: E402
from data_generator import ensure_dataset  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SECTIONS = ["Executive Overview", "Campaign Management", "Advertising Creatives", "Big Data Analytics",
            "Framework Benchmarking", "Budget Optimization", "Sentiment Analytics", "Dataset Management",
            "About / Documentation"]


@pytest.fixture(scope="module")
def populated(tmp_path_factory):
    root = tmp_path_factory.mktemp("dashboard")
    patch = pytest.MonkeyPatch()
    patch.setattr(store, "DATA_DIR", root / "data")
    patch.setattr(store, "CACHE_DIR", root / "cache")
    patch.setattr(store, "RESULTS_DIR", root / "results")
    path = ensure_dataset(30_000, root / "data", seed=5)
    store.RESULTS_DIR.mkdir(parents=True)
    run_benchmarks({30_000: path}, ("pandas",), trials=2).to_csv(store.RESULTS_DIR / "benchmark_results.csv", index=False)
    yield root
    patch.undo()


@pytest.fixture
def app(populated):
    return streamlit_testing.AppTest.from_file(APP, default_timeout=120).run()


def go(app, section):
    app.sidebar.radio(key="nav").set_value(section).run()
    return app


def labels(app):
    return [m.label for m in app.metric]


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_renders_without_errors(app, section):
    go(app, section)
    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_overview_shows_measured_kpis(app):
    go(app, "Executive Overview")
    assert {"Ad spend", "Revenue", "ROAS", "CTR", "CPA"} <= set(labels(app))
    assert any("simulated" in c.value.lower() for c in app.caption)
    charts = [c.proto.spec for c in app.get("plotly_chart")]
    assert any("CTR by platform" in c for c in charts) and any("Spend vs revenue by platform" in c for c in charts)


def test_platform_filter_changes_totals(app):
    go(app, "Executive Overview")
    full = {m.label: m.value for m in app.metric}
    app.sidebar.multiselect(key="platform_filter").set_value(["Google"]).run()
    assert not app.exception
    filtered = {m.label: m.value for m in app.metric}
    assert filtered["Ad spend"] != full["Ad spend"]


def test_campaign_management_lists_campaigns_and_details(app):
    go(app, "Campaign Management")
    assert {"Budget (simulated)", "ROAS", "Conversions"} <= set(labels(app))
    assert app.dataframe and "campaign_id" in app.dataframe[0].value.columns


def test_creative_page_labels_creatives_as_simulated(app):
    go(app, "Advertising Creatives")
    assert any("simulated" in w.value.lower() for w in app.warning)
    assert {"ROAS", "CTR", "CPC", "CPA", "CPM"} <= set(labels(app))


def test_big_data_page_reports_processing_and_quality(app):
    go(app, "Big Data Analytics")
    assert {"Records read", "Valid records", "Chunks"} <= set(labels(app))
    assert "Missing key fields" in labels(app)


def test_benchmark_page_shows_measured_results(app):
    go(app, "Framework Benchmarking")
    assert any(not df.value.empty for df in app.dataframe)
    assert any("Measured values only" in c.value for c in app.caption)


@pytest.mark.parametrize("level,expected_column", [("Platform", "platform"), ("Campaign", "campaign_id")])
def test_budget_optimization_runs_at_both_levels(app, level, expected_column):
    go(app, "Budget Optimization")
    app.radio(key="opt_level").set_value(level).run()
    [b for b in app.button if b.label == "Optimise"][0].click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert any("not guaranteed" in m.value.lower() for m in app.markdown)
    tables = [df.value for df in app.dataframe if "reason" in df.value.columns]
    assert tables and expected_column in tables[0].columns


def test_budget_optimization_reports_infeasible_limits(app):
    go(app, "Budget Optimization")
    app.radio(key="opt_level").set_value("Platform").run()
    app.slider[0].set_value((30, 100))
    [b for b in app.button if b.label == "Optimise"][0].click().run()
    assert not app.exception
    assert any("No recommendation" in e.value for e in app.error)


def test_sentiment_page_without_upload_shows_empty_state(app):
    go(app, "Sentiment Analytics")
    assert any("No sentiment data loaded" in i.value for i in app.info)


def test_dataset_management_lists_generated_dataset(app):
    go(app, "Dataset Management")
    assert app.dataframe and 30_000 in app.dataframe[0].value["Records"].tolist()


def test_missing_data_directory_shows_helpful_empty_states(tmp_path):
    patch = pytest.MonkeyPatch()
    patch.setattr(store, "DATA_DIR", tmp_path / "nothing")
    try:
        app = streamlit_testing.AppTest.from_file(APP, default_timeout=60).run()
        go(app, "Executive Overview")
        assert not app.exception
        assert any("No data to show" in i.value for i in app.info)
        go(app, "Budget Optimization")
        assert not app.exception and any("No dataset selected" in i.value for i in app.info)
    finally:
        patch.undo()
