"""Command-line entry point that runs every pipeline stage in order."""

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from field_ops import config, evaluate, excel_report, generate, libreoffice, metrics, quality
from field_ops.classify import classify_reports
from field_ops.clean import clean_reports

CLEAN_CSV = config.OUTPUT_DIR / "clean_reports.csv"
DQ_REPORT = config.OUTPUT_DIR / "data_quality.md"
DQ_TABLES_DIR = "data_quality"
EXCEL_REPORT = config.OUTPUT_DIR / "field_ops_report.xlsx"
METRICS_REPORT = config.OUTPUT_DIR / "metrics_report.md"
METRICS_DIR = config.OUTPUT_DIR / "metrics"
VENDORS_CSV = config.REFERENCE_DIR / "vendors.csv"


def run_clean(
    raw_path: Path = config.RAW_REPORTS,
    output_dir: Path = config.OUTPUT_DIR,
    truth_dir: Path = config.TRUTH_DIR,
) -> quality.DataQualityReport:
    """Clean + classify the raw export; write the clean CSV and the data-quality report."""
    # keep_default_na=False: read blanks as "" and never turn text such as "NA" into NaN.
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    result = clean_reports(raw)
    classified = classify_reports(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    classified.final.to_csv(output_dir / CLEAN_CSV.name, index=False)

    report = quality.build_report(result, raw, evaluate.load_truth(truth_dir), classified)
    quality.write_report(report, output_dir / DQ_REPORT.name)
    quality.write_tables(report, output_dir / DQ_TABLES_DIR)
    s = report.summary
    print(
        f"Cleaned {s['raw_rows']} raw rows -> {s['final_rows']} tickets "
        f"({s['exact_duplicates_removed']} exact + {s['near_duplicates_removed']} near dups); "
        f"categories: {report.classification['source_counts']}; "
        f"{CLEAN_CSV.relative_to(config.ROOT)}, {DQ_REPORT.relative_to(config.ROOT)}"
    )
    return report


def run_metrics(
    clean_csv: Path = CLEAN_CSV,
    output_dir: Path = config.OUTPUT_DIR,
    truth_dir: Path = config.TRUTH_DIR,
    repeat_window: int = config.REPEAT_WINDOW_DAYS,
) -> metrics.Metrics:
    """Compute the operational metrics from the clean CSV (the contract with Excel/dashboard)."""
    clean = metrics.load_clean_csv(clean_csv)
    vendors = metrics.load_vendors() if VENDORS_CSV.exists() else None
    m = metrics.compute_all(clean, config.AS_OF, repeat_window, vendors)

    tables_dir = output_dir / METRICS_DIR.name
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in m.tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    scenario = evaluate.load_scenario(truth_dir)
    checks = evaluate.planted_pattern_checks(m, scenario) if scenario else None
    (output_dir / METRICS_REPORT.name).write_text(metrics.to_markdown(m, checks))
    k = m.kpis
    print(
        f"Metrics: median MTTR {k['median_days']:.1f}d (mean {k['mean_days']:.1f}d), "
        f"{k['open']} open ({k['open_over_30_days']} > 30d), repeat rate {k['repeat_rate']:.1%}; "
        f"{METRICS_REPORT.relative_to(config.ROOT)}"
    )
    return m


def run_excel(
    clean_csv: Path = CLEAN_CSV,
    output_dir: Path = config.OUTPUT_DIR,
    repeat_window: int = config.REPEAT_WINDOW_DAYS,
) -> excel_report.Layout:
    """Build the Excel report (formulas over the clean data) from the clean CSV."""
    clean = metrics.load_clean_csv(clean_csv)
    vendors = metrics.load_vendors()
    m = metrics.compute_all(clean, config.AS_OF, repeat_window, vendors)
    dq = excel_report.load_dq(output_dir / DQ_TABLES_DIR)
    path = output_dir / EXCEL_REPORT.name
    layout = excel_report.build_workbook(m, vendors, path, dq)
    print(f"Excel report: {layout.n_rows} tickets -> {path.relative_to(config.ROOT)}")
    return layout


def run_verify_excel(path: Path = EXCEL_REPORT) -> int:
    """Recalculate the delivered workbook in LibreOffice and fail on any formula error."""
    found = libreoffice.find_soffice()
    if found:  # print the version first, so a failure below is easy to attribute
        print(f"verify-excel: found {libreoffice.version_string(found)} at {found}")
    try:
        soffice = libreoffice.locate()
    except libreoffice.LibreOfficeUnavailable as problem:
        if libreoffice.required():
            raise SystemExit(f"verify-excel: {problem}") from problem
        print(f"verify-excel skipped: {problem}")
        return 0
    print(f"verify-excel: using {libreoffice.version_string(soffice)}")
    with tempfile.TemporaryDirectory() as tmp:
        recalculated = libreoffice.recalculate(path, Path(tmp), soffice)
        errors = libreoffice.find_errors(recalculated)
    if errors:
        shown = ", ".join(f"{sheet}!{cell} {value}" for sheet, cell, value in errors[:10])
        raise SystemExit(f"verify-excel: {len(errors)} formula errors, e.g. {shown}")
    print(f"verify-excel: {path.relative_to(config.ROOT)} recalculates with no formula errors")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="field-ops", description=__doc__)
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["generate", "clean", "metrics", "excel", "verify-excel", "all"],
    )
    parser.add_argument(
        "--repeat-window",
        type=int,
        default=config.REPEAT_WINDOW_DAYS,
        help="days after a fault closes within which a new same-system/category fault is a repeat",
    )
    args = parser.parse_args(argv)

    if args.step in ("generate", "all"):
        generate.main()
    if args.step in ("clean", "all"):
        run_clean()
    if args.step in ("metrics", "all"):
        run_metrics(repeat_window=args.repeat_window)
    if args.step in ("excel", "all"):
        run_excel(repeat_window=args.repeat_window)
    if args.step in ("verify-excel", "all"):
        run_verify_excel()


if __name__ == "__main__":
    main()
