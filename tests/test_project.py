import numpy as np
import pandas as pd
import pytest
from budget_optimizer import BudgetOptimizer
from data_generator import generate_dataframe
from metrics_calculator import compute_kpis, aggregate_csv

def test_generator_reproducible():
    pd.testing.assert_frame_equal(generate_dataframe(100, 7), generate_dataframe(100, 7))

def test_zero_records_has_schema(tmp_path):
    from data_generator import generate_csv, COLUMNS
    path = generate_csv(0, tmp_path / "empty.csv")
    assert list(pd.read_csv(path).columns) == COLUMNS

def test_kpi_formulas():
    df = generate_dataframe(500, 10)
    out = compute_kpis(df)
    assert np.isclose(out.loc[0, "ctr_pct"], df.clicks.sum()/df.impressions.sum()*100)
    assert np.isclose(out.loc[0, "roas"], df.revenue.sum()/df.spend.sum())

def test_optimizer_valid_budget_and_constraints():
    dates = pd.date_range("2025-01-01", periods=30)
    rows=[]
    for p, mult in [("A", 1.2), ("B", .9), ("C", .7)]:
        for i, date in enumerate(dates):
            spend=100+3*i
            rows.append({"platform":p,"date":date,"spend":spend,"revenue":mult*spend**.8})
    daily=pd.DataFrame(rows)
    opt=BudgetOptimizer(.05,.8).fit(daily)
    result=opt.recommend(1000)
    assert np.isclose(result.table.recommended_daily_spend.sum(),1000,atol=1e-5)
    assert (result.table.recommended_share_pct >= 5-1e-5).all()
    assert (result.table.recommended_share_pct <= 80+1e-5).all()

def test_optimizer_rejects_bad_budget():
    dates=pd.date_range("2025-01-01",periods=20)
    daily=pd.DataFrame([{"platform":p,"date":d,"spend":100+i,"revenue":(100+i)**.8}
      for p in ["A","B","C"] for i,d in enumerate(dates)])
    opt=BudgetOptimizer(.05,.8).fit(daily)
    for bad in [0,-1,np.nan,np.inf]:
        with pytest.raises(ValueError):
            opt.recommend(bad)

def test_aggregate_empty_csv(tmp_path):
    from data_generator import generate_csv
    path=generate_csv(0,tmp_path/"empty.csv")
    result=aggregate_csv(path,["platform","date"])
    assert result.empty
