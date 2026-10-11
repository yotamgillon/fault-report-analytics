from types import SimpleNamespace

import pandas as pd
import pytest

from field_ops import evaluate, quality
from field_ops.classify import classify_reports
from field_ops.clean import clean_reports
from field_ops.generate import generate_dataset, write_dataset


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory):
    """Generate, clean, classify and score the seeded dataset once for the whole test session."""
    base = tmp_path_factory.mktemp("pipeline")
    write_dataset(generate_dataset(), raw_path=base / "raw.csv", truth_dir=base / "truth")
    raw = pd.read_csv(base / "raw.csv", dtype=str, keep_default_na=False)
    truth = evaluate.load_truth(base / "truth")
    scenario = evaluate.load_scenario(base / "truth")
    result = clean_reports(raw)
    classified = classify_reports(result)
    report = quality.build_report(result, raw, truth, classified)
    return SimpleNamespace(
        raw=raw,
        truth=truth,
        scenario=scenario,
        result=result,
        classified=classified,
        report=report,
    )


@pytest.fixture(scope="session")
def soffice():
    """Path to a LibreOffice new enough for XLOOKUP. CI sets FIELD_OPS_REQUIRE_LIBREOFFICE=1 so a
    missing LibreOffice fails the build there instead of silently skipping the formula checks."""
    from field_ops import libreoffice

    try:
        return libreoffice.locate()
    except libreoffice.LibreOfficeUnavailable as problem:
        if libreoffice.required():
            pytest.fail(f"LibreOffice is required but unusable: {problem}")
        pytest.skip(f"LibreOffice not available: {problem}")


def pytest_sessionstart(session):
    """Print the LibreOffice version first thing, even with -q (the formula checks depend on it)."""
    from field_ops import libreoffice

    soffice = libreoffice.find_soffice()
    line = (
        f"LibreOffice: {libreoffice.version_string(soffice)} ({soffice})"
        if soffice
        else "LibreOffice: not found (Excel formula checks skipped unless required)"
    )
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(line)


@pytest.fixture(scope="session")
def metrics_run(pipeline_run):
    """Metrics (with the demo SLA table) over the seeded dataset's clean output."""
    from field_ops import config, metrics

    return metrics.compute_all(
        pipeline_run.classified.final,
        config.AS_OF,
        config.REPEAT_WINDOW_DAYS,
        metrics.load_vendors(),
    )
