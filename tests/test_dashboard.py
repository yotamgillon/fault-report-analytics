"""The dashboard: its numbers come from metrics.dashboard(), and the app itself does no maths.

Part 1 tests metrics.dashboard() against an independent pandas oracle (tests/helpers.py).
Part 2 drives the Streamlit app headlessly (streamlit.testing.v1.AppTest).
Part 3 checks the app source for business logic.
"""

import ast
from pathlib import Path

import helpers
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from field_ops import config, metrics

APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
KPI_KEYS = {  # metric card label -> (key in metrics KPIs, key in the pandas oracle)
    "Open faults": ("open", "open"),
    "MTTR, median (days)": ("median_days", "median"),
    "Repeat rate": ("repeat_rate", "repeat"),
    "SLA breach (demo)": ("sla_breach_rate", "breach"),
    "Tickets": ("tickets", "tickets"),
}
SELECTIONS = [
    {"vendors": ["Vendor F"], "sites": ["Ridgeview"]},
    {"vendors": ["Vendor A", "Vendor B"], "severities": ["High", "Critical"]},
    {"sites": ["Lakeside"], "severities": ["Low"]},
    {"severities": ["Critical"]},
]


def as_filters(selection: dict) -> metrics.Filters:
    return metrics.Filters(
        vendors=tuple(selection["vendors"]) if "vendors" in selection else None,
        sites=tuple(selection["sites"]) if "sites" in selection else None,
        severities=tuple(selection["severities"]) if "severities" in selection else None,
    )


# --- Part 1: metrics.dashboard() ---------------------------------------------------------------
def test_unfiltered_dashboard_equals_the_overall_metrics(metrics_run):
    view = metrics.dashboard(metrics_run)
    assert view.kpis == metrics_run.kpis
    assert view.fleet == metrics_run.kpis
    base = [c for c in metrics.SCORECARD_COLUMNS if not c.endswith("_pct")]
    pd.testing.assert_frame_equal(
        view.scorecard[base].reset_index(drop=True),
        metrics_run.scorecard[base].reset_index(drop=True),
    )
    pct = view.scorecard
    assert ((pct["repeat_rate_pct"] - pct["repeat_rate"] * 100).abs() < 1e-9).all()
    assert ((pct["sla_breach_rate_pct"] - pct["sla_breach_rate"] * 100).abs() < 1e-9).all()


@pytest.mark.parametrize("selection", SELECTIONS)
def test_filtered_kpis_match_an_independent_pandas_calculation(metrics_run, selection):
    view = metrics.dashboard(metrics_run, as_filters(selection))
    want = helpers.expected_kpis(
        metrics_run, selection.get("vendors"), selection.get("sites"), selection.get("severities")
    )
    got = {
        "tickets": view.kpis["tickets"], "resolved": view.kpis["resolved"],
        "open": view.kpis["open"], "open30": view.kpis["open_over_30_days"],
        "median": view.kpis["median_days"], "mean": view.kpis["mean_days"],
        "breach": view.kpis["sla_breach_rate"], "repeat": view.kpis["repeat_rate"],
    }  # fmt: skip
    for key, expected in want.items():
        if expected is None:
            assert pd.isna(got[key]), key
        else:
            assert got[key] == pytest.approx(expected), key


def test_repeat_flags_come_from_the_whole_dataset_not_the_filtered_subset(metrics_run):
    """Filtering by severity must not change which faults count as repeats of an earlier fault."""
    selection = {"severities": ["Critical"]}
    view = metrics.dashboard(metrics_run, as_filters(selection))
    mask = metrics_run.prepared["severity"] == "Critical"
    flagged_on_all = int(metrics_run.repeat.rows.loc[mask, "is_repeat"].sum())
    assert view.kpis["repeats"] == flagged_on_all
    recomputed = metrics.repeat_faults(metrics_run.prepared[mask]).summary["repeats"]
    assert recomputed != flagged_on_all  # recomputing on the subset would give a different answer
    assert view.fleet["repeat_rate"] == metrics_run.kpis["repeat_rate"]  # fleet is not filtered


def test_scorecard_marks_low_sample_vendors_and_ranks_adequate_ones_first(metrics_run):
    view = metrics.dashboard(metrics_run, as_filters({"severities": ["Critical"]}))
    card = view.scorecard
    assert (card["low_sample"] == (card["resolved_n"] < metrics.MIN_SAMPLE)).all()
    assert card["low_sample"].any() and not card["low_sample"].all()
    flags = card["low_sample"].tolist()
    assert flags == sorted(flags)  # every adequate vendor before every low-sample one
    adequate = card.loc[~card["low_sample"], "median_days"]
    assert adequate.is_monotonic_decreasing


def test_scorecard_carries_the_demo_sla_columns(metrics_run):
    card = metrics.dashboard(metrics_run).scorecard.set_index("vendor")
    vendors = metrics.load_vendors().set_index("vendor")
    assert (card["sla_target_days"] == vendors["sla_target_days"].reindex(card.index)).all()
    assert (card["tier"] == vendors["tier"].reindex(card.index)).all()
    assert (card["sla_breaches"] <= card["resolved_n"]).all()


def test_top_systems_sorted_by_repeat_rate_with_fleet_comparison(metrics_run):
    top = metrics.dashboard(metrics_run).top_systems
    assert len(top) == 10 and top["rank"].tolist() == list(range(1, 11))
    assert top["repeat_rate"].is_monotonic_decreasing
    fleet = metrics_run.kpis["repeat_rate"]
    assert (top["vs_fleet_x"] - top["repeat_rate"] / fleet).abs().max() < 1e-9
    assert (top["counted_faults"] >= metrics.MIN_SAMPLE).all()


def test_monthly_and_aging_follow_the_selection(metrics_run):
    selection = {"sites": ["Ridgeview"]}
    view = metrics.dashboard(metrics_run, as_filters(selection))
    sub = metrics_run.prepared[metrics_run.prepared["site"] == "Ridgeview"]
    assert view.monthly["opened"].sum() == len(sub)
    assert view.aging["tickets"].sum() == int(sub["is_open"].sum())
    assert view.aging["bucket"].tolist() == metrics_run.aging_overall.columns[2:6].tolist()


def test_empty_selection_returns_empty_tables_with_the_same_columns(metrics_run):
    view = metrics.dashboard(metrics_run, metrics.Filters(vendors=()))
    assert view.empty and view.kpis["tickets"] == 0
    assert list(view.scorecard.columns) == metrics.SCORECARD_COLUMNS and view.scorecard.empty
    assert list(view.monthly.columns) == metrics.MONTHLY_COLUMNS and view.monthly.empty
    assert list(view.top_systems.columns) == metrics.TOP_COLUMNS
    assert view.aging["tickets"].sum() == 0


def test_filter_options_cover_every_value_in_canonical_order(metrics_run):
    options = metrics.filter_options(metrics_run)
    assert options["severities"] == list(config.SEVERITIES)
    assert len(options["vendors"]) == 15 and len(options["sites"]) == 6


def test_definitions_state_the_repeat_window_in_use():
    text = " ".join(f"{t} {d}" for t, d in metrics.definitions(21))
    assert "21 days" in text and "demo" in text


# --- Part 2: the Streamlit app, run headlessly -------------------------------------------------
@pytest.fixture(scope="session")
def clean_csv(pipeline_run, tmp_path_factory):
    path = tmp_path_factory.mktemp("app") / "clean_reports.csv"
    pipeline_run.classified.final.to_csv(path, index=False)
    return path


@pytest.fixture
def run_app(monkeypatch, clean_csv):
    def run(csv=None, **filters) -> AppTest:
        monkeypatch.setenv("FIELD_OPS_CLEAN_CSV", str(csv or clean_csv))
        at = AppTest.from_file(str(APP), default_timeout=120).run()
        names = {
            "vendors": "vendor_filter",
            "sites": "site_filter",
            "severities": "severity_filter",
        }
        for field, values in filters.items():
            at.sidebar.multiselect(key=names[field]).set_value(values)
        return at.run() if filters else at

    return run


def parse(text: str) -> float:
    cleaned = text.replace(",", "")
    return float(cleaned.rstrip("%")) / 100 if cleaned.endswith("%") else float(cleaned)


def cards(at: AppTest) -> dict[str, tuple[str, str]]:
    return {m.label: (m.value, m.delta) for m in at.metric}


def test_app_renders_without_errors(run_app):
    at = run_app()
    assert not at.exception and not at.error
    assert set(cards(at)) == set(KPI_KEYS)
    assert len(at.dataframe) == 3  # performance and SLA (demo) tables, top repeat offenders


def test_app_kpis_equal_the_metrics_for_all_tickets(run_app, metrics_run):
    shown = cards(run_app())
    for label, (kpi_key, _) in KPI_KEYS.items():
        want = metrics_run.kpis[kpi_key]
        tolerance = 5e-4 if kpi_key.endswith("rate") else 0.05
        assert parse(shown[label][0]) == pytest.approx(want, abs=tolerance), label


@pytest.mark.parametrize("selection", SELECTIONS)
def test_app_filters_match_an_independent_pandas_calculation(run_app, metrics_run, selection):
    shown = cards(run_app(**selection))
    want = helpers.expected_kpis(
        metrics_run, selection.get("vendors"), selection.get("sites"), selection.get("severities")
    )
    for label, (_, oracle_key) in KPI_KEYS.items():
        expected = want[oracle_key]
        value = shown[label][0]
        if expected is None:
            assert value == "–", label
        else:
            tolerance = 5e-4 if label in ("Repeat rate", "SLA breach (demo)") else 0.05
            assert parse(value) == pytest.approx(expected, abs=tolerance), (label, selection)


def test_sla_numbers_are_labelled_as_demo_values(run_app):
    at = run_app()
    sla_labels = [label for label in cards(at) if "SLA" in label]
    assert sla_labels and all("demo" in label for label in sla_labels)
    assert any("demo values" in c.value for c in at.caption)
    assert any("demo SLA targets" in c.value for c in at.caption)  # the card's own explanation
    columns = [c for frame in at.dataframe for c in frame.value.columns]
    sla_columns = [c for c in columns if "SLA" in c or "Tier" in c]
    assert sla_columns and all("demo" in c for c in sla_columns)
    assert not any("SLA" in c for c in at.dataframe[0].value.columns)  # performance table
    assert [t.label for t in at.tabs] == ["Performance", "SLA (demo values)"]


def test_app_shows_n_next_to_the_median_and_in_the_scorecard(run_app, metrics_run):
    at = run_app()
    _, delta = cards(at)["MTTR, median (days)"]
    assert f"n = {metrics_run.kpis['resolved']:,}" in delta
    table = at.dataframe[0].value
    assert {"Resolved (n)", "Sample", "Median days", "Mean days"} <= set(table.columns)
    assert list(table.columns).index("Median days") < list(table.columns).index("Mean days")
    assert {"Resolved (n)", "Sample"} <= set(at.dataframe[1].value.columns)  # n + flag on SLA too
    assert "Faults counted (n)" in at.dataframe[2].value.columns


def test_app_flags_low_sample_groups_and_selections(run_app, metrics_run):
    at = run_app(severities=["Critical"])
    table = at.dataframe[0].value
    assert (table["Sample"] == "low sample").any() and (table["Sample"] == "").any()
    assert (table.loc[table["Sample"] == "low sample", "Resolved (n)"] < 10).all()
    # a selection that is itself too small gets the same warning as the Excel dashboard
    p = metrics_run.prepared
    small = p[p["is_resolved"]].groupby(["vendor", "site", "severity"]).size()
    vendor, site, severity = small[small < metrics.MIN_SAMPLE].index[0]
    at = run_app(vendors=[vendor], sites=[site], severities=[severity])
    assert [w.value for w in at.warning if "Low sample" in w.value]
    assert "fewer than 10 resolved tickets" in at.warning[0].value


def test_app_handles_an_empty_selection(run_app):
    at = run_app(vendors=[])
    assert not at.exception
    assert any("No tickets match" in i.value for i in at.info)
    assert len(at.metric) == 0


def test_app_explains_a_missing_data_file(run_app, tmp_path):
    at = run_app(csv=tmp_path / "missing.csv")
    assert not at.exception
    assert any("make all" in e.value for e in at.error)
    assert len(at.metric) == 0


# --- Part 3: no business logic in the app ------------------------------------------------------
CALCULATION_METHODS = {
    "groupby", "median", "mean", "sum", "merge", "resample", "agg", "aggregate", "rolling",
    "pivot", "pivot_table", "value_counts", "count", "quantile", "std", "var", "cumsum",
    "nunique", "rank", "corr", "diff", "pct_change", "isin", "query", "sort_values",
}  # fmt: skip
ALLOWED_IMPORTS = {"pathlib", "pandas", "plotly", "streamlit", "field_ops"}


def test_app_imports_only_presentation_libraries_and_the_package():
    tree = ast.parse(APP.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= ALLOWED_IMPORTS, imported - ALLOWED_IMPORTS
    from_package = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "field_ops"
        for alias in node.names
    }
    assert from_package == {"config", "metrics"}  # no reaching into clean/classify/evaluate


def test_app_does_not_calculate_anything_itself():
    tree = ast.parse(APP.read_text())
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (called & CALCULATION_METHODS), called & CALCULATION_METHODS
    assert "metrics" in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    uses_dashboard = any(
        isinstance(n, ast.Attribute) and n.attr == "dashboard" for n in ast.walk(tree)
    )
    assert uses_dashboard


def test_the_check_for_calculations_can_fail():
    snippet = ast.parse("df.groupby('vendor')['days'].median()")
    called = {
        n.func.attr for n in ast.walk(snippet)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }  # fmt: skip
    assert called & CALCULATION_METHODS == {"groupby", "median"}
