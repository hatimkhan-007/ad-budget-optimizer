import main as pipeline


def test_pipeline_creates_clear_charts(tmp_path):
    charts = tmp_path / "charts"
    pipeline.main(["--scales", "20000", "--engines", "pandas", "--trials", "1",
                   "--data-dir", str(tmp_path / "data"), "--results-dir", str(tmp_path / "results"),
                   "--charts-dir", str(charts)])
    files = sorted(charts.glob("*.png"))
    assert 5 <= len(files) <= 6
    assert all(f.stat().st_size > 10_000 for f in files)


def test_skip_benchmark_creates_five_charts(tmp_path):
    charts = tmp_path / "charts"
    pipeline.main(["--scales", "10000", "--skip-benchmark", "--data-dir", str(tmp_path / "d"),
                   "--results-dir", str(tmp_path / "r"), "--charts-dir", str(charts)])
    assert len(list(charts.glob("*.png"))) == 5
