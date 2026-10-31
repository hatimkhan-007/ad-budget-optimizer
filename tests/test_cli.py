import pandas as pd
import pytest

import benchmark_analysis
import main as pipeline


def test_pipeline_end_to_end(tmp_path):
    pipeline.main(["--scales", "20000", "--engines", "pandas", "mapreduce", "--trials", "1",
                   "--data-dir", str(tmp_path / "data"), "--results-dir", str(tmp_path / "results"),
                   "--campaign-budget", "500000", "--currency", "PKR"])
    results = tmp_path / "results"
    budget = pd.read_csv(results / "budget_recommendation.csv")
    assert abs(budget["recommended_share_pct"].sum() - 100) < 1e-6
    assert abs(budget["recommended_campaign_amount"].sum() - 500_000) < 1e-3
    assert (budget["recommended_share_pct"] >= 5 - 1e-6).all() and (budget["recommended_share_pct"] <= 40 + 1e-6).all()
    assert set(pd.read_csv(results / "kpis_by_platform.csv")["platform"]) == set(budget["platform"])
    assert (pd.read_csv(results / "benchmark_results.csv")["consistent"] == "yes").all()
    assert len(pd.read_csv(results / "budget_summary.csv")) == 1


def test_skip_benchmark_writes_no_benchmark_file(tmp_path):
    pipeline.main(["--scales", "10000", "--skip-benchmark", "--data-dir", str(tmp_path / "d"),
                   "--results-dir", str(tmp_path / "r")])
    assert not (tmp_path / "r" / "benchmark_results.csv").exists()


@pytest.mark.parametrize("argv", [["--scales", "0"], ["--trials", "0"], ["--budget", "-5"], ["--campaign-budget", "0"]])
def test_main_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit):
        pipeline.parse_args(argv)


def test_benchmark_cli(tmp_path):
    benchmark_analysis.main(["--scales", "10000", "--engines", "pandas", "--trials", "1",
                             "--data-dir", str(tmp_path / "d"), "--results-dir", str(tmp_path / "r")])
    assert (tmp_path / "r" / "benchmark_results.csv").exists()
