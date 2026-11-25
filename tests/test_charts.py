import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from budget_optimizer import BudgetOptimizer
from charts import chart_roas, chart_spend_vs_revenue, generate_charts
from metrics_calculator import compute_kpis


@pytest.fixture(scope="module")
def chart_inputs(large_frame):
    daily = large_frame.assign(date=pd.to_datetime(large_frame["date"])).groupby(["platform", "date"], observed=True)[
        ["spend", "revenue"]].sum().reset_index()
    optimizer = BudgetOptimizer(0.05, 0.40).fit(daily)
    return compute_kpis(large_frame, "platform"), daily, optimizer, optimizer.recommend().table


def test_individual_charts_write_png_files(tmp_path, chart_inputs):
    kpis = chart_inputs[0]
    for function, name in [(chart_roas, "roas.png"), (chart_spend_vs_revenue, "spend.png")]:
        path = function(kpis, tmp_path / name)
        assert path.exists() and path.stat().st_size > 1_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_charts_creates_numbered_set_and_replaces_old_ones(tmp_path, chart_inputs):
    kpis, daily, optimizer, table = chart_inputs
    (tmp_path / "99_old.png").write_bytes(b"stale")
    made = generate_charts(tmp_path, kpis, daily, optimizer, table, 0.05, 0.40)
    assert len(made) >= 5 and all(p.exists() for p in made)
    assert not (tmp_path / "99_old.png").exists()
