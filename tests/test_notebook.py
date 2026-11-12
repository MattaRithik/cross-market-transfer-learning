"""Check notebook science and cache reuse with isolated synthetic fixtures."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Cross_Market_Transfer_Learning.ipynb"


def notebook_functions():
    namespace = {}
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and "definitions" in cell.metadata.get("tags", []):
            exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    namespace.update(
        TARGET="ret_exc_lead1m",
        KEY=["excntry", "id", "eom"],
        COUNTRIES=["USA", "GBR", "JPN"],
        DOWNLOAD_START="1999-12-31",
        candidates=["signal", "noise"],
        manifest={"end": "2023-12-31"},
        COST_BPS=[0, 10, 25],
        feature_names=["signal", "noise"],
        DEVICE=namespace["torch"].device("cpu"),
        MAX_EPOCHS=2,
        PATIENCE=2,
        RUN_MODE="smoke",
        BOOTSTRAPS=99,
        CPU_THREADS=1,
    )
    namespace["torch"].set_num_threads(1)
    return namespace


FUNCTIONS = notebook_functions()
data = SimpleNamespace(**FUNCTIONS)


@pytest.fixture(scope="module")
def ns():
    return FUNCTIONS


def sample_panel(country="GBR", months=24, start="2016-12-31", stocks=12):
    rng = np.random.default_rng(7)
    dates = pd.date_range(start, periods=months, freq="ME")
    frame = pd.MultiIndex.from_product([dates, range(stocks)], names=["eom", "id"]).to_frame(
        index=False
    )
    frame["excntry"] = country
    frame["id"] = frame.id.astype(str)
    frame["signal"] = rng.uniform(-1, 1, len(frame))
    frame["noise"] = rng.uniform(-1, 1, len(frame))
    frame["ret_exc_lead1m"] = frame.signal * 0.01 + rng.normal(0, 0.02, len(frame))
    frame["target_month"] = frame.eom + pd.offsets.MonthEnd(1)
    frame["ret"] = 0.01
    frame["size_grp"] = "small"
    frame["me"] = 100.0
    for flag in data.SCREENS:
        frame[flag] = 1
    return frame


def test_return_alignment_and_splits(ns):
    frame = sample_panel(months=72, start="2015-12-31")
    cleaned, _ = ns["clean_country"](frame, "GBR")
    assert cleaned.iloc[0].target_month == pd.Timestamp("2016-01-31")
    december = cleaned.loc[cleaned.eom.eq("2019-12-31")]
    assert ns["partition"](december, "test").target_month.eq("2020-01-31").all()
    one_year = ns["partition"](cleaned, "train", 1)
    assert one_year.target_month.min() == pd.Timestamp("2017-01-31")
    assert one_year.target_month.max() == pd.Timestamp("2017-12-31")
    split = [set(ns["partition"](cleaned, p).target_month) for p in ["train", "validation", "test"]]
    assert all(not a & b for i, a in enumerate(split) for b in split[i + 1 :])
    with pytest.raises(AssertionError):
        ns["clean_country"](pd.concat([frame, frame.iloc[:1]]), "GBR")


def test_ranking_and_missing_labels(ns):
    f = sample_panel(months=1, stocks=4)
    f["signal"] = [1.0, 3.0, 2.0, np.nan]
    ranked = ns["rank_numerical"](f, ["signal"])
    np.testing.assert_allclose(ranked.signal, [-0.5, 0.5, 0, 0])
    f.loc[0, "ret_exc_lead1m"] = np.nan
    pd.testing.assert_frame_equal(ranked, ns["rank_numerical"](f, ["signal"]))
    only = ns["rank_numerical"](f.iloc[:1], ["signal"])
    assert only.signal.iloc[0] == 0
    # Categories retain semantic labels and learn imputation/vocabulary from training only.
    modes, encoder = ns["fit_categories"](
        pd.DataFrame({"sector": ["a", "a", "b", None]}), ["sector"]
    )
    assert modes == {"sector": "a"}
    assert encoder.transform(pd.DataFrame({"sector": ["new"]})).sum() == 0


def test_training_only_predictors(ns):
    frames = {c: sample_panel(c, months=36, start="2014-12-31") for c in data.COUNTRIES}
    meta = pd.DataFrame({"abr_jkp": ["signal", "noise"], "name": ["Known signal", "Noise"]})
    for c in ["GBR", "JPN"]:
        mask = frames[c].target_month.dt.year.eq(2017)
        frames[c].loc[mask, "noise"] = np.nan
    selected, audit = ns["choose_predictors"](frames, meta)
    assert selected == ["signal"]
    changed = {c: f.copy() for c, f in frames.items()}
    for f in changed.values():
        f.loc[f.target_month.dt.year.lt(2017) & f.excntry.ne("USA"), "noise"] = 999
        f["ret_exc_lead1m"] = np.nan
    assert ns["choose_predictors"](changed, meta)[0] == selected
    assert "ret_exc_lead1m" not in selected


def test_transformer_equivariance_variable_stocks_and_gradients(ns):
    torch = ns["torch"]
    torch.manual_seed(19)
    model = ns["CrossStockTransformer"](2, width=64, blocks=2, dropout=0.1).eval()
    x = torch.randn(13, 2)
    order = torch.randperm(13)
    with torch.no_grad():
        np.testing.assert_allclose(
            model(x[order]).numpy(), model(x)[order].numpy(), atol=2e-6, rtol=1e-5
        )
        assert model(torch.randn(7, 2)).shape == (7,)
        # Cross-stock interaction: changing a peer changes another stock's forecast.
        changed = x.clone()
        changed[1] += 10
        assert not torch.isclose(model(x)[0], model(changed)[0], atol=1e-8)
    model.train()
    model(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_metrics_ensemble_and_month_weighting(ns):
    f = sample_panel(months=2, stocks=10)
    f["ret_exc_lead1m"] = 1.0
    prediction = ns["prediction_frame"](f, np.zeros(len(f)))
    metrics = ns["predictive_metrics"](prediction)
    assert metrics["mse_stock_weighted"] == 1 and metrics["r2_zero_stock_weighted"] == 0
    perfect = ns["prediction_frame"](f, np.ones(len(f)))
    assert ns["predictive_metrics"](perfect)["r2_zero_month_weighted"] == 1
    ensemble = ns["equal_average"]([prediction, perfect])
    np.testing.assert_allclose(ensemble.prediction, 0.5)
    broken = perfect.iloc[1:]
    with pytest.raises(AssertionError):
        ns["equal_average"]([prediction, broken])
    unequal = pd.concat([f.iloc[:10], f.iloc[10:12]], ignore_index=True)
    weights = ns["month_weights"](unequal)
    assert weights[:10].sum() == pytest.approx(weights[10:].sum())


def test_portfolios_costs_missing_returns_and_drift(ns):
    f = sample_panel(months=2, stocks=10)
    f["ret_exc_lead1m"] = np.tile(np.arange(10) / 100, 2)
    predictions = ns["prediction_frame"](f, np.tile(np.arange(10), 2))
    weights = ns["decile_weights"](predictions.iloc[:10])
    assert weights["9"] == 1 and weights["0"] == -1
    backtest = ns["portfolio_returns"](predictions, f)
    np.testing.assert_allclose(backtest.gross, 0.09)
    assert backtest.traded_notional.iloc[0] == 2
    # Second rebalance trades .01 in each leg after 1% total-return drift; plus liquidation.
    assert backtest.traded_notional.iloc[1] == pytest.approx(2.02)
    np.testing.assert_allclose(
        backtest.net_10bps, backtest.gross - 0.001 * backtest.traded_notional
    )
    assert ns["traded_notional"](weights, weights) == 0
    missing = predictions.copy()
    missing.loc[missing.id.eq("9"), "ret_exc_lead1m"] = np.nan
    result = ns["portfolio_returns"](missing, f)
    assert result.gross.isna().all() and result.missing_holdings.eq(1).all()


def test_bootstrap_pairing_and_calendar_gaps(ns):
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    a, b = (
        pd.Series(np.arange(24) / 100, index=dates),
        pd.Series(np.arange(24) / 100 - 0.01, index=dates),
    )
    report = ns["paired_bootstrap"](a, b, repetitions=99)
    assert report["difference"] == pytest.approx(0.01)
    assert report["ci_low"] == pytest.approx(0.01)
    assert report["p_value"] == 0.01
    a = a.drop(dates[11])
    assert ns["paired_bootstrap"](a, b, repetitions=99)["months"] == 23
    idx = ns["moving_block_indices"](a.index, 20, np.random.default_rng(0))
    ordinals = a.index.to_period("M").astype(int).to_numpy()
    for row in idx:
        for offset in range(0, 18, 6):
            assert np.diff(ordinals[row[offset : offset + 6]]).tolist() == [1] * 5


class MockWRDS:
    def __init__(self):
        self.calls = []

    def describe_table(self, **kwargs):
        return pd.DataFrame({"name": data.REQUIRED + ["signal", "ret"]})

    def raw_sql(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if "MAX(eom)" in query:
            return pd.DataFrame({"endpoint": [pd.Timestamp("2000-12-31")]})
        params = kwargs["params"]
        f = sample_panel(params["country"], months=1, start=params["lower"], stocks=10)
        return iter([f[data.REQUIRED + ["signal", "ret"]]])


def test_download_once_resume_and_no_network_cache_reads(tmp_path):
    db = MockWRDS()
    manifest = data.initialize_snapshot(db, tmp_path, ["signal"], "1999-12-31")
    _, missing = data.cache_status(tmp_path)
    data.download_missing(db, tmp_path, manifest, missing)
    count = len(db.calls)
    manifest, missing = data.cache_status(tmp_path, verify_hash=True)
    assert not missing
    data.download_missing(db, tmp_path, manifest, missing)
    assert len(db.calls) == count, "A rerun must issue NO SQL requests"
    assert len(data.load_country(tmp_path, "GBR")) == 20
    missing_path = tmp_path / "JPN/2000.parquet"
    missing_path.unlink()
    manifest, missing = data.cache_status(tmp_path)
    assert missing == ["JPN/2000.parquet"]
    data.download_missing(db, tmp_path, manifest, missing)
    assert len(db.calls) == count + 1
    assert not data.cache_status(tmp_path, verify_hash=True)[1]
    assert not list(tmp_path.rglob("*.partial"))
    with pytest.raises(ValueError):
        data.validate_snapshot(manifest, ["changed"], "1999-12-31", "contrib.global_factor")


def test_local_import_offline(tmp_path):
    source, target = tmp_path / "source", tmp_path / "raw"
    source.mkdir()
    target.mkdir()
    for c in data.COUNTRIES:
        sample_panel(c, months=2, start="1999-12-31").to_parquet(
            source / f"{c}.parquet", index=False
        )
    data.register_local_files(target, source, ["signal"], "1999-12-31")
    assert not data.cache_status(target)[1]
    assert len(data.load_country(target, "USA")) == 24


def test_checkpoints_resume_and_cached_training(ns, tmp_path):
    # Definitions use the original globals dictionary, so update it for the isolated paths.
    ns.update(MODELS=tmp_path, RESULTS=tmp_path, SEEDS=[11])
    (tmp_path / "training").mkdir()
    train = sample_panel(months=2, stocks=10)
    validation = sample_panel(months=2, start="2018-01-31", stocks=10)
    params = dict(width=64, blocks=2, dropout=0.1, lr=0.001, weight_decay=0.0001)
    network, report = ns["fit_network"](train, validation, params, 11, "test")
    expected = ns["predict_network"](network, validation)
    path = tmp_path / "test.pt"
    before = path.stat().st_mtime_ns
    loaded, _ = ns["fit_network"](train, validation, params, 11, "test")
    np.testing.assert_array_equal(ns["predict_network"](loaded, validation), expected)
    assert before == path.stat().st_mtime_ns
    # Reconstruct an interrupted first epoch and verify exact resumed results.
    ns["MAX_EPOCHS"] = 1
    ns["fit_network"](train, validation, params, 11, "resume")
    checkpoint = ns["torch"].load(tmp_path / "resume.pt", weights_only=False)
    checkpoint["complete"] = False
    ns["torch"].save(checkpoint, tmp_path / "resume.pt")
    ns["MAX_EPOCHS"] = 2
    resumed, _ = ns["fit_network"](train, validation, params, 11, "resume")
    np.testing.assert_array_equal(ns["predict_network"](resumed, validation), expected)


def test_wrds_sql_reader_dependency_compatibility():
    """An offline SQL read exercises actual WRDS/Pandas/SQLAlchemy integration."""
    import wrds
    from sqlalchemy import create_engine

    connection = wrds.Connection(autoconnect=False)
    connection.engine = create_engine("sqlite://")
    connection.connection = connection.engine.connect()
    try:
        chunks = list(connection.raw_sql("SELECT 1 AS value", chunksize=1, return_iter=True))
        assert chunks[0].value.iloc[0] == 1
    finally:
        connection.close()


def test_credential_cell_and_inspector_initialization(ns, monkeypatch):
    import sqlalchemy
    import wrds

    calls = {}

    class FakeConnection:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs
            self.connection = object()

        def connect_directly(self, raise_err):
            calls["raise_err"] = raise_err

    FakeConnection._Connection__make_sa_engine_conn = FakeConnection.connect_directly
    monkeypatch.setattr(wrds, "Connection", FakeConnection)
    monkeypatch.setattr(sqlalchemy, "inspect", lambda connection: "inspector ready")
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)
    answers = iter(["fixture_username", "fixture_password"])
    namespace = {
        "DATA_METHOD": "wrds",
        "missing_parts": ["one partition"],
        "os": ns["os"],
        "getpass": lambda _: next(answers),
    }
    notebook = nbformat.read(ROOT / "Cross_Market_Transfer_Learning.ipynb", as_version=4)
    cell = next(
        c for c in notebook.cells if c.cell_type == "code" and "WRDS username (hidden)" in c.source
    )
    exec(cell.source, namespace)
    assert calls["kwargs"]["wrds_password"] == "fixture_password"
    assert calls["kwargs"]["autoconnect"] is False and calls["raise_err"] is True
    assert namespace["wrds_connection"].insp == "inspector ready"
    assert "wrds_username" not in namespace and "wrds_password" not in namespace


def test_shared_forecast_tags(ns):
    assert ns["prediction_tag"]("GBR", 1, "zero_shot", 11) == ns["prediction_tag"](
        "GBR", 5, "zero_shot", 11
    )
    assert ns["prediction_tag"]("GBR", 1, "local", 11) != ns["prediction_tag"](
        "GBR", 5, "local", 11
    )


def test_complete_notebook_and_cached_rerun(tmp_path):
    """Run all experiments twice; forbid login/network and check raw/model cache reuse."""
    project = tmp_path / "project"
    metadata_dir = project / "data/metadata"
    metadata_dir.mkdir(parents=True)
    imports = project / "data/local_import"
    imports.mkdir()
    raw = project / "data/raw"
    raw.mkdir()
    shutil.copy(NOTEBOOK, project / NOTEBOOK.name)
    # Official names, synthetic values. These fixtures are never research observations.
    names = ["ret_12_1", "be_me", "at_gr1", "ni_me"]
    pd.DataFrame({"abr_jkp": names, "name": names, "group": "Synthetic test fixture"}).to_csv(
        metadata_dir / "factor_details.csv", index=False
    )
    rng = np.random.default_rng(431)
    for country in data.COUNTRIES:
        frame = sample_panel(country, months=265, start="1999-12-31", stocks=12)
        for name in names:
            frame[name] = rng.normal(size=len(frame))
        frame.loc[frame.eom.eq(frame.eom.max()), "ret_exc_lead1m"] = np.nan
        frame.to_parquet(imports / f"{country}.parquet", index=False)
    data.register_local_files(raw, imports, names, "1999-12-31")
    assert not data.cache_status(raw, verify_hash=True)[1]

    spec = tmp_path / "kernels/finance-test"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Notebook validation",
                "language": "python",
                "env": {
                    "RUN_MODE": "smoke",
                    "DATA_METHOD": "wrds",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "MPLBACKEND": "Agg",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            }
        )
    )
    manager = KernelSpecManager(kernel_dirs=[str(spec.parent)])
    guard = nbformat.v4.new_code_cell("""import getpass, requests, wrds

def forbidden(*args, **kwargs):
    raise AssertionError("A complete cache must not request credentials or connect to the network")
getpass.getpass = forbidden
requests.get = forbidden
wrds.Connection = forbidden
""")

    def timestamps(directory, pattern):
        return {str(p.relative_to(project)): p.stat().st_mtime_ns for p in directory.rglob(pattern)}

    raw_before = timestamps(raw, "*.parquet")
    model_before = None
    for _ in range(2):
        notebook = nbformat.read(project / NOTEBOOK.name, as_version=4)
        notebook.cells.insert(0, guard)
        client = NotebookClient(
            notebook,
            timeout=120,
            kernel_name="finance-test",
            resources={"metadata": {"path": str(project)}},
        )
        client.km = KernelManager(kernel_name="finance-test", kernel_spec_manager=manager)
        client.execute()
        assert timestamps(raw, "*.parquet") == raw_before
        models = timestamps(project / "models", "*.pt")
        if model_before is not None:
            assert models == model_before, "Completed models must not be retrained on a rerun"
        model_before = models

    outputs = list((project / "results/smoke").glob("*/tables"))
    assert len(outputs) == 1
    tables = outputs[0]
    assert all(list(tables.glob(f"table_{number}_*")) for number in range(1, 9))
    assert len(pd.read_csv(tables / "table_4_predictive_performance.csv")) == 42
    assert len(pd.read_csv(tables / "table_7_significance.csv")) == 630
    assert (
        pd.read_csv(tables / "table_8_research_findings.csv")
        .finding.str.contains("SMOKE DIAGNOSTIC")
        .all()
    )
    # One U.S. tuning fit plus local, adaptation, and pooled fits for six experiments.
    assert len(model_before) == 19
    predictions = tables.parent / "predictions"
    assert len(list(predictions.glob("*_zero_shot_seed*_test.parquet"))) == 2
