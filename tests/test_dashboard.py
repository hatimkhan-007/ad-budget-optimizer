import shutil
from pathlib import Path

import pytest

import main as pipeline

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_dashboard_renders_generated_results(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    pipeline.main(["--scales", "10000", "--engines", "pandas", "--trials", "1",
                   "--data-dir", str(tmp_path / "data"), "--results-dir", str(tmp_path / "results")])
    shutil.copy(project / "app.py", tmp_path / "app.py")
    monkeypatch.chdir(tmp_path)
    app = streamlit_testing.AppTest.from_file(str(tmp_path / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert not app.error
    assert len(app.tabs) == 5
    labels = [m.label for m in app.metric]
    assert "Total ad spend" in labels and "Uplift vs limit-compliant baseline" in labels
