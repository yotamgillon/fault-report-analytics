"""Operational metrics computed from the delivered clean CSV.

Everything here takes the clean report table (the pipeline's contract with the Excel report and
the dashboard), so a number can never differ between them.

Definitions, stated once:
- Resolved ticket: status Closed with a valid close date >= open date, and none of the
  dq_flags closed_no_close_date / close_before_open / date_invalid. Only resolved tickets enter
  resolution-time (MTTR) numbers.
- Open ticket: status Open or In Progress. Age is measured against the fixed AS_OF date.
- Repeat fault: same system_id and same category, opened on or after the *previous* such fault's
  close date and within ``window_days`` (default 14) of it. A fault opened while the previous one
  was still open is not a repeat (see ``repeat_faults`` for the edge cases).
- A group with fewer than MIN_SAMPLE resolved tickets is marked low-sample and never ranked, so a
  vendor with 8 tickets cannot look like the worst (or the best) on noise.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from field_ops import config
from field_ops.markdown import md_table, pct

MIN_SAMPLE = 10
REPEAT_WINDOW_DAYS = config.REPEAT_WINDOW_DAYS
OPEN_STATUSES = ("Open", "In Progress")
UNRESOLVED_FLAGS = ("closed_no_close_date", "close_before_open", "date_invalid")
# (label, first day, last day); the last bucket is open-ended.
AGE_BUCKETS = [
    ("0-7 days", 0, 7),
    ("8-30 days", 8, 30),
    ("31-90 days", 31, 90),
    ("91+ days", 91, np.inf),
]
MTTR_GROUPS = {
    "vendor": "vendor",
    "site": "site",
    "system": "system_id",
    "system type": "system_type",
}


def load_vendors(path=config.REFERENCE_DIR / "vendors.csv") -> pd.DataFrame:
    """Vendor reference data: tier, SLA target (days), support region, contract type."""
    return pd.read_csv(path, dtype={"sla_target_days": int})


def has_flag(flags: pd.Series, name: str) -> pd.Series:
    """True where the semicolon-separated dq_flags contain exactly ``name``."""
    return (";" + flags.fillna("") + ";").str.contains(f";{name};", regex=False)


def prepare(df: pd.DataFrame, as_of: date = config.AS_OF) -> pd.DataFrame:
    """Parse dates and derive the resolved / open / age columns used everywhere else."""
    out = df.copy()
    for col in ("date_opened", "date_closed"):
        out[col] = pd.to_datetime(out[col], format="%Y-%m-%d", errors="coerce")
    out["dq_flags"] = out["dq_flags"].fillna("")

    unusable = pd.Series(False, index=out.index)
    for flag in UNRESOLVED_FLAGS:
        unusable |= has_flag(out["dq_flags"], flag)
    days = (out["date_closed"] - out["date_opened"]).dt.days
    out["is_resolved"] = (out["status"] == "Closed") & (days >= 0) & ~unusable
    out["days_to_resolve"] = days.where(out["is_resolved"])
    out["is_open"] = out["status"].isin(OPEN_STATUSES) & out["date_opened"].notna()
    age = (pd.Timestamp(as_of) - out["date_opened"]).dt.days.clip(lower=0)
    out["age_days"] = age.where(out["is_open"])
    return out


def add_sla(prep: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    """Join each ticket to its vendor's SLA target; a resolved ticket slower than it breaches."""
    out = prep.copy()
    out["sla_target_days"] = out["vendor"].map(vendors.set_index("vendor")["sla_target_days"])
    out["sla_breach"] = (
        out["is_resolved"] & out["sla_target_days"].notna()
        & (out["days_to_resolve"] > out["sla_target_days"])
    )  # fmt: skip
    return out


def sla_by_vendor(prep: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    """Reference columns per vendor plus SLA breaches among resolved tickets."""
    resolved = prep[prep["is_resolved"] & prep["sla_target_days"].notna()]
    breaches = resolved.groupby("vendor")["sla_breach"].agg(sla_checked="size", sla_breaches="sum")
    table = vendors.set_index("vendor").join(breaches).fillna({"sla_checked": 0, "sla_breaches": 0})
    table["sla_breach_rate"] = (table["sla_breaches"] / table["sla_checked"]).where(
        table["sla_checked"] > 0
    )
    return table.reset_index().astype({"sla_checked": int, "sla_breaches": int})


# ---------------------------------------------------------------------------------------------
# Resolution time
# ---------------------------------------------------------------------------------------------
def mttr_by(prep: pd.DataFrame, column: str, min_sample: int = MIN_SAMPLE) -> pd.DataFrame:
    """Median first, mean beside it, n per group; groups under ``min_sample`` are not ranked."""
    tickets = prep.groupby(column).size().rename("tickets")
    resolved = (
        prep[prep["is_resolved"]]
        .groupby(column)["days_to_resolve"]
        .agg(n="size", median_days="median", mean_days="mean")
    )
    table = tickets.to_frame().join(resolved)
    table["n"] = table["n"].fillna(0).astype(int)
    table["low_sample"] = table["n"] < min_sample
    adequate = table["median_days"].where(~table["low_sample"])
    table["rank_slowest"] = adequate.rank(ascending=False, method="min")
    table = table.assign(_sort=table["median_days"].fillna(-1)).sort_values(
        ["low_sample", "_sort"], ascending=[True, False], kind="stable"
    )
    return table.drop(columns="_sort").reset_index()


# ---------------------------------------------------------------------------------------------
# Repeat faults
# ---------------------------------------------------------------------------------------------
@dataclass
class RepeatResult:
    rows: pd.DataFrame  # indexed like the prepared table: counted, is_repeat, overlap, exclusion
    summary: dict


def repeat_faults(prep: pd.DataFrame, window_days: int = REPEAT_WINDOW_DAYS) -> RepeatResult:
    """Flag faults that came back on the same system in the same category.

    A fault is a repeat when it opens on or after the close date of the previous same-system,
    same-category fault and no more than ``window_days`` later. Edge cases, decided up front:
    - opened on the close date itself: after-close, so it can be a repeat (gap of 0 days).
    - opened *before* the previous fault closed, or while it is still open: NOT a repeat. It stays
      in the denominator and is counted as ``opened_while_previous_open`` so the volume is visible.
    - previous fault closed but its close date is unknown: cannot be judged, so the row is left
      out of numerator and denominator and counted in ``not_evaluable``.
    - Unclassified/Unknown categories and rows flagged category_text_mismatch are excluded
      entirely (also as the "previous" fault): their category cannot be trusted.
    """
    df = prep
    reason = pd.Series("", index=df.index)
    reason[~df["fault_category"].isin(config.CATEGORIES)] = "unclassified_category"
    mismatch = has_flag(df["dq_flags"], "category_text_mismatch")
    reason[(reason == "") & mismatch] = "category_text_mismatch"
    reason[(reason == "") & df["date_opened"].isna()] = "no_open_date"

    eligible = df[reason == ""].sort_values(
        ["system_id", "fault_category", "date_opened", "report_id"], kind="stable"
    )
    grouped = eligible.groupby(["system_id", "fault_category"], sort=False)
    has_prev = grouped.cumcount() > 0
    prev_resolved = grouped["is_resolved"].shift().fillna(False).astype(bool)
    prev_open = grouped["is_open"].shift().fillna(False).astype(bool)
    gap = (eligible["date_opened"] - grouped["date_closed"].shift()).dt.days

    is_repeat = has_prev & prev_resolved & (gap >= 0) & (gap <= window_days)
    opened_while_open = has_prev & ((prev_resolved & (gap < 0)) | prev_open)
    not_evaluable = has_prev & ~prev_resolved & ~prev_open
    reason.loc[eligible.index[not_evaluable]] = "previous_close_unknown"

    counted_mask = reason == ""
    rows = pd.DataFrame(
        {
            "counted": counted_mask,
            "is_repeat": is_repeat.reindex(df.index, fill_value=False) & counted_mask,
            "overlap": opened_while_open.reindex(df.index, fill_value=False) & counted_mask,
            "exclusion": reason,
        }
    )
    counted = int(rows["counted"].sum())
    repeats = int(rows["is_repeat"].sum())
    summary = {
        "window_days": window_days,
        "total_rows": len(df),
        "excluded_unclassified": int((reason == "unclassified_category").sum()),
        "excluded_text_mismatch": int((reason == "category_text_mismatch").sum()),
        "excluded_no_open_date": int((reason == "no_open_date").sum()),
        "not_evaluable": int((reason == "previous_close_unknown").sum()),
        "denominator": counted,
        "repeats": repeats,
        "opened_while_previous_open": int(rows["overlap"].sum()),
        "repeat_rate": repeats / counted if counted else float("nan"),
    }
    return RepeatResult(rows=rows, summary=summary)


def repeat_sensitivity(
    prep: pd.DataFrame,
    windows: tuple[int, ...] = config.SENSITIVITY_WINDOWS,
    configured: int | None = None,
) -> pd.DataFrame:
    """Overall repeat rate for several windows (the configured one is always included).

    If the rate swings widely with the window, it is measuring how busy a system is rather
    than whether faults come back, so the choice of window matters and should be visible.
    """
    rows = []
    for window in sorted({*windows, *([configured] if configured is not None else [])}):
        result = repeat_faults(prep, window)
        rows.append(
            {
                "window_days": window,
                "counted_faults": result.summary["denominator"],
                "repeats": result.summary["repeats"],
                "repeat_rate": result.summary["repeat_rate"],
            }
        )
    return pd.DataFrame(rows)


def repeat_by(
    prep: pd.DataFrame, repeat: RepeatResult, column: str, min_sample: int = MIN_SAMPLE
) -> pd.DataFrame:
    """Repeat rate per group, compared with the fleet average (pooled: all repeats / all faults).

    ``vs_fleet_pts`` is the difference in percentage points, ``vs_fleet_x`` the ratio
    (1.0 = same as the fleet).
    """
    counted = repeat.rows["counted"]
    table = (
        pd.DataFrame({column: prep[column], "repeat": repeat.rows["is_repeat"]})[counted]
        .groupby(column)["repeat"]
        .agg(counted_faults="size", repeats="sum")
    )
    fleet = repeat.summary["repeat_rate"]
    table["repeat_rate"] = table["repeats"] / table["counted_faults"]
    table["fleet_rate"] = fleet
    table["vs_fleet_pts"] = (table["repeat_rate"] - fleet) * 100
    table["vs_fleet_x"] = table["repeat_rate"] / fleet if fleet else np.nan
    table["low_sample"] = table["counted_faults"] < min_sample
    return table.reset_index()


def repeat_by_system(
    prep: pd.DataFrame, repeat: RepeatResult, min_sample: int = MIN_SAMPLE
) -> pd.DataFrame:
    """Every system with its repeat rate vs the fleet, plus vendor, site and resolution time."""
    by_system = repeat_by(prep, repeat, "system_id", min_sample)
    info = prep.groupby("system_id").agg(
        vendor=("vendor", "first"), site=("site", "first"), tickets=("report_id", "size")
    )
    mttr = mttr_by(prep, "system_id", min_sample).set_index("system_id")[["median_days", "n"]]
    table = by_system.join(info, on="system_id").join(mttr, on="system_id")
    table = table.sort_values(
        ["low_sample", "repeat_rate", "repeats", "system_id"],
        ascending=[True, False, False, True],
    )
    return table.rename(columns={"n": "resolved_n"}).reset_index(drop=True)


def top_repeat_offenders(
    prep: pd.DataFrame, repeat: RepeatResult, n: int = 10, min_sample: int = MIN_SAMPLE
) -> pd.DataFrame:
    """Systems ranked by repeat rate (ties: more repeats first); low-sample systems are left out."""
    table = repeat_by_system(prep, repeat, min_sample)
    table = table[~table["low_sample"]].copy()
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Backlog aging
# ---------------------------------------------------------------------------------------------
def _bucket_counts(ages: pd.Series) -> dict:
    return {label: int(((ages >= lo) & (ages <= hi)).sum()) for label, lo, hi in AGE_BUCKETS}


def aging(prep: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Open tickets bucketed by age against AS_OF, overall or per group."""
    groups = [("All", prep)] if by is None else list(prep.groupby(by))
    rows = []
    for name, sub in groups:
        ages = sub.loc[sub["is_open"], "age_days"]
        counts = _bucket_counts(ages)
        rows.append(
            {
                (by or "scope"): name,
                "open": len(ages),
                **counts,
                "over_30_days": counts["31-90 days"] + counts["91+ days"],
                "median_age_days": float(ages.median()) if len(ages) else float("nan"),
                "oldest_days": int(ages.max()) if len(ages) else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Monthly trend
# ---------------------------------------------------------------------------------------------
def monthly_trend(
    prep: pd.DataFrame, as_of: date = config.AS_OF, min_sample: int = MIN_SAMPLE
) -> pd.DataFrame:
    """Opened vs closed per month, resolution time of tickets closed that month, and the
    backlog at each month end (tickets closed with an unknown date are left out of the backlog)."""
    opened = prep["date_opened"].dropna()
    if opened.empty:
        return pd.DataFrame()
    months = pd.period_range(opened.min().to_period("M"), pd.Timestamp(as_of).to_period("M"))
    resolved = prep[prep["is_resolved"]]
    closed_month = resolved["date_closed"].dt.to_period("M")
    rows = []
    for month in months:
        month_end = min(month.end_time.normalize(), pd.Timestamp(as_of))
        in_month = resolved[closed_month == month]["days_to_resolve"]
        still_open = prep["is_open"] | (prep["is_resolved"] & (prep["date_closed"] > month_end))
        rows.append(
            {
                "month": str(month),
                "opened": int((prep["date_opened"].dt.to_period("M") == month).sum()),
                "closed": len(in_month),
                "median_days": float(in_month.median()) if len(in_month) else float("nan"),
                "mean_days": float(in_month.mean()) if len(in_month) else float("nan"),
                "open_at_month_end": int((still_open & (prep["date_opened"] <= month_end)).sum()),
                "low_sample": len(in_month) < min_sample,
            }
        )
    table = pd.DataFrame(rows)
    table.insert(3, "net_opened", table["opened"] - table["closed"])
    return table


# ---------------------------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------------------------
@dataclass
class Metrics:
    prepared: pd.DataFrame
    as_of: date
    kpis: dict
    mttr: dict[str, pd.DataFrame]
    repeat: RepeatResult
    repeat_sensitivity: pd.DataFrame
    repeat_by_vendor: pd.DataFrame
    repeat_by_system: pd.DataFrame
    top_offenders: pd.DataFrame
    aging_overall: pd.DataFrame
    aging_by_site: pd.DataFrame
    aging_by_vendor: pd.DataFrame
    monthly: pd.DataFrame
    scorecard: pd.DataFrame
    vendors: pd.DataFrame | None
    tables: dict[str, pd.DataFrame]


def vendor_scorecard(
    mttr: pd.DataFrame,
    repeat_vendor: pd.DataFrame,
    aging_vendor: pd.DataFrame,
    sla: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per vendor, in the same order as the MTTR table (low-sample vendors last)."""
    card = mttr.merge(
        repeat_vendor[["vendor", "counted_faults", "repeats", "repeat_rate", "vs_fleet_x"]],
        on="vendor",
        how="left",
    ).merge(aging_vendor[["vendor", "open", "over_30_days"]], on="vendor", how="left")
    if sla is not None:
        card = card.merge(sla, on="vendor", how="left")
    return card.rename(columns={"n": "resolved_n"})


def compute_all(
    df: pd.DataFrame,
    as_of: date = config.AS_OF,
    repeat_window: int = REPEAT_WINDOW_DAYS,
    vendors: pd.DataFrame | None = None,
) -> Metrics:
    prep = prepare(df, as_of)
    if vendors is not None:
        prep = add_sla(prep, vendors)
    repeat = repeat_faults(prep, repeat_window)
    mttr = {name: mttr_by(prep, col) for name, col in MTTR_GROUPS.items()}
    repeat_vendor = repeat_by(prep, repeat, "vendor")
    aging_vendor = aging(prep, "vendor")
    aging_site = aging(prep, "site")
    sla = sla_by_vendor(prep, vendors) if vendors is not None else None
    kpi_values = kpis(prep, repeat.rows)
    scorecard = vendor_scorecard(mttr["vendor"], repeat_vendor, aging_vendor, sla)
    repeat_system = repeat_by_system(prep, repeat)
    top = top_repeat_offenders(prep, repeat)
    sensitivity = repeat_sensitivity(prep, configured=repeat_window)
    monthly = monthly_trend(prep, as_of)
    aging_all = aging(prep)
    tables = {
        **{f"mttr_by_{name.replace(' ', '_')}": t for name, t in mttr.items()},
        "repeat_by_vendor": repeat_vendor,
        "repeat_sensitivity": sensitivity,
        "repeat_by_system": repeat_system,
        "top_repeat_offenders": top,
        "aging_overall": aging_all,
        "aging_by_site": aging_site,
        "aging_by_vendor": aging_vendor,
        "monthly_trend": monthly,
        "vendor_scorecard": scorecard,
    }
    return Metrics(
        prepared=prep,
        as_of=as_of,
        kpis=kpi_values,
        mttr=mttr,
        repeat=repeat,
        repeat_sensitivity=sensitivity,
        repeat_by_vendor=repeat_vendor,
        repeat_by_system=repeat_system,
        top_offenders=top,
        aging_overall=aging_all,
        aging_by_site=aging_site,
        aging_by_vendor=aging_vendor,
        monthly=monthly,
        scorecard=scorecard,
        vendors=vendors,
        tables=tables,
    )


# ---------------------------------------------------------------------------------------------
# Headline numbers for any selection of tickets (used by compute_all and the dashboard)
# ---------------------------------------------------------------------------------------------
def kpis(prep: pd.DataFrame, repeat_rows: pd.DataFrame, min_sample: int = MIN_SAMPLE) -> dict:
    """Headline numbers for the tickets in ``prep`` (all of them, or a filtered selection).

    ``repeat_rows`` must come from ``repeat_faults`` on the *whole* dataset, restricted to the same
    tickets: whether a fault is a repeat depends on the previous fault, which a filter may hide.
    """
    resolved = prep.loc[prep["is_resolved"], "days_to_resolve"]
    counted = int(repeat_rows["counted"].sum())
    repeats = int(repeat_rows["is_repeat"].sum())
    has_sla = "sla_breach" in prep
    checked = int((prep["is_resolved"] & prep["sla_target_days"].notna()).sum()) if has_sla else 0
    breaches = int(prep["sla_breach"].sum()) if has_sla else 0
    nan = float("nan")
    return {
        "tickets": len(prep),
        "resolved": len(resolved),
        "open": int(prep["is_open"].sum()),
        "open_over_30_days": int((prep["age_days"] > 30).sum()),
        "median_days": float(resolved.median()) if len(resolved) else nan,
        "mean_days": float(resolved.mean()) if len(resolved) else nan,
        "repeat_counted": counted,
        "repeats": repeats,
        "repeat_rate": repeats / counted if counted else nan,
        "sla_checked": checked,
        "sla_breaches": breaches,
        "sla_breach_rate": breaches / checked if checked else nan,
        "low_sample": len(resolved) < min_sample,
    }


# ---------------------------------------------------------------------------------------------
# Dashboard view: filters in, ready-to-render tables out
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Filters:
    """None = no restriction on that field; a tuple keeps only those values (empty = nothing)."""

    vendors: tuple[str, ...] | None = None
    sites: tuple[str, ...] | None = None
    severities: tuple[str, ...] | None = None


# *_pct columns repeat the rate as percentage points (0-100), ready for table display.
SCORECARD_COLUMNS = [
    "vendor", "tickets", "resolved_n", "median_days", "mean_days", "low_sample", "repeat_rate",
    "vs_fleet_x", "open", "over_30_days", "tier", "sla_target_days", "sla_breaches",
    "sla_breach_rate", "repeat_rate_pct", "sla_breach_rate_pct",
]  # fmt: skip
MONTHLY_COLUMNS = [
    "month", "opened", "closed", "net_opened", "median_days", "mean_days", "open_at_month_end",
    "low_sample",
]  # fmt: skip
TOP_COLUMNS = [
    "rank", "system_id", "vendor", "site", "counted_faults", "repeats", "repeat_rate",
    "vs_fleet_pts", "vs_fleet_x", "tickets", "median_days", "resolved_n", "repeat_rate_pct",
]  # fmt: skip


def definitions(repeat_window: int = REPEAT_WINDOW_DAYS) -> list[tuple[str, str]]:
    """Plain-language definitions shown next to the numbers (term, explanation)."""
    return [
        ("Resolved", "Status Closed with a usable close date. Only resolved tickets enter MTTR."),
        ("MTTR", "Mean time to resolve. The median is shown first (resolution times are "
                 "right-skewed); the mean sits beside it."),
        ("Low sample", f"Fewer than {MIN_SAMPLE} resolved tickets: do not rank or compare on it."),
        ("Open", "Status Open or In Progress. Age is measured from the open date to the fixed "
                 "reporting date."),
        ("Repeat fault", f"Same system and category opening within {repeat_window} days after "
                         "the previous fault was closed. A fault opened while the previous one "
                         "was still open is not a repeat. Unclassified or text-mismatch "
                         "categories are excluded."),
        ("Fleet", "All tickets, regardless of the filters."),
        ("SLA (demo)", "Targets come from data/reference/vendors.csv: synthetic demo values, "
                       "not real contract terms."),
    ]  # fmt: skip


@dataclass
class Dashboard:
    filters: Filters
    as_of: date
    repeat_window: int
    min_sample: int
    n_total: int
    kpis: dict
    fleet: dict  # the same numbers for every ticket, for context
    scorecard: pd.DataFrame
    monthly: pd.DataFrame
    aging: pd.DataFrame  # bucket, tickets
    top_systems: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.kpis["tickets"] == 0


def filter_options(m: Metrics) -> dict[str, list[str]]:
    severities = [v for v in config.SEVERITIES if v in set(m.prepared["severity"])]
    return {
        "vendors": sorted(m.prepared["vendor"].unique()),
        "sites": sorted(m.prepared["site"].unique()),
        "severities": severities,
    }


def _selection_mask(prep: pd.DataFrame, filters: Filters) -> pd.Series:
    mask = pd.Series(True, index=prep.index)
    for column, allowed in (
        ("vendor", filters.vendors),
        ("site", filters.sites),
        ("severity", filters.severities),
    ):
        if allowed is not None:
            mask &= prep[column].isin(allowed)
    return mask


def aging_buckets(prep: pd.DataFrame) -> pd.DataFrame:
    """Open tickets per age bucket, one row per bucket (long form, ready to chart)."""
    row = aging(prep).iloc[0]
    return pd.DataFrame({"bucket": [b[0] for b in AGE_BUCKETS],
                         "tickets": [int(row[b[0]]) for b in AGE_BUCKETS]})  # fmt: skip


def dashboard(m: Metrics, filters: Filters | None = None) -> Dashboard:
    """Everything the dashboard shows for a filter selection. The app renders; it never computes.

    Repeat flags and the fleet rate come from the whole dataset (see ``kpis``); every other
    number is recomputed from the selected tickets only.
    """
    filters = filters or Filters()
    mask = _selection_mask(m.prepared, filters)
    prep, rows = m.prepared[mask], m.repeat.rows[mask]
    repeat = RepeatResult(rows=rows, summary=m.repeat.summary)  # fleet rate stays global
    fleet = kpis(m.prepared, m.repeat.rows)
    base = {
        "filters": filters, "as_of": m.as_of, "repeat_window": m.repeat.summary["window_days"],
        "min_sample": MIN_SAMPLE, "n_total": len(m.prepared), "kpis": kpis(prep, rows),
        "fleet": fleet,
    }  # fmt: skip
    if prep.empty:
        return Dashboard(
            **base,
            scorecard=pd.DataFrame(columns=SCORECARD_COLUMNS),
            monthly=pd.DataFrame(columns=MONTHLY_COLUMNS),
            aging=pd.DataFrame({"bucket": [b[0] for b in AGE_BUCKETS], "tickets": 0}),
            top_systems=pd.DataFrame(columns=TOP_COLUMNS),
        )
    sla = sla_by_vendor(prep, m.vendors) if m.vendors is not None else None
    card = vendor_scorecard(
        mttr_by(prep, "vendor"),
        repeat_by(prep, repeat, "vendor"),
        aging(prep, "vendor"),
        sla,
    )
    card = card.assign(
        repeat_rate_pct=card["repeat_rate"] * 100,
        sla_breach_rate_pct=card.get("sla_breach_rate", float("nan")) * 100,
    )
    top = top_repeat_offenders(prep, repeat)
    return Dashboard(
        **base,
        scorecard=card.reindex(columns=SCORECARD_COLUMNS),
        monthly=monthly_trend(prep, m.as_of).reindex(columns=MONTHLY_COLUMNS),
        aging=aging_buckets(prep),
        top_systems=top.assign(repeat_rate_pct=top["repeat_rate"] * 100).reindex(
            columns=TOP_COLUMNS
        ),
    )


def load_clean_csv(path=None) -> pd.DataFrame:
    """The pipeline's clean CSV, read the way every consumer must read it (all text, no NA)."""
    return pd.read_csv(path or config.clean_csv_path(), dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------
def _show(df: pd.DataFrame) -> pd.DataFrame:
    """Round floats and print booleans / NaN readably for Markdown."""
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda v: "–" if pd.isna(v) else f"{v:.1f}")
        elif shown[col].dtype == bool:
            shown[col] = shown[col].map(lambda v: "low sample" if v else "")
    return shown.rename(columns={"low_sample": "sample"})


def _rate_table(df: pd.DataFrame) -> pd.DataFrame:
    shown = df.drop(columns=["fleet_rate"], errors="ignore")
    shown["repeat_rate"] = shown["repeat_rate"].map(pct)
    shown["vs_fleet_pts"] = shown["vs_fleet_pts"].map(lambda v: f"{v:+.1f} pts")
    shown["vs_fleet_x"] = shown["vs_fleet_x"].map(lambda v: "–" if pd.isna(v) else f"{v:.2f}x")
    return shown


def _sens_table(df: pd.DataFrame, used: int) -> pd.DataFrame:
    shown = df.copy()
    shown["repeat_rate"] = shown["repeat_rate"].map(pct)
    shown["window_days"] = shown["window_days"].map(
        lambda w: f"{w}" + (" (used)" if w == used else "")
    )
    return shown


def to_markdown(m: Metrics, checks: pd.DataFrame | None = None) -> str:
    k, r = m.kpis, m.repeat.summary
    out = [
        "# Operational metrics",
        "",
        f"Measured against the fixed reporting date **{m.as_of}**. Groups with fewer than "
        f"{MIN_SAMPLE} resolved tickets are marked *low sample* and are not ranked.",
        "",
        "## Headline numbers",
        "",
        f"- Tickets: **{k['tickets']}**, resolved (usable close date): **{k['resolved']}**, "
        f"open: **{k['open']}** ({k['open_over_30_days']} older than 30 days)",
        f"- Resolution time: median **{k['median_days']:.1f}** days, mean "
        f"**{k['mean_days']:.1f}** days (median first: the distribution is right-skewed)",
        f"- Repeat rate: **{pct(k['repeat_rate'])}** "
        f"({r['repeats']} of {r['denominator']} counted faults)",
        "",
        "## Repeat faults",
        "",
        "A repeat is a fault on the same system and category that opens **after** the previous "
        f"fault was closed and within **{r['window_days']} days** of its close date. A fault "
        "opened while the previous one was still open is not a repeat "
        f"({r['opened_while_previous_open']} such faults: counted, but not as repeats).",
        "",
        "Fleet average = all repeats / all counted faults. `vs_fleet_pts` is a system's rate minus "
        "the fleet rate in percentage points; `vs_fleet_x` is the ratio (1.00x = fleet average).",
        "",
        f"Excluded from numerator **and** denominator ({r['total_rows'] - r['denominator']} of "
        f"{r['total_rows']} rows):",
        "",
        f"- {r['excluded_unclassified']} rows with an Unclassified/Unknown category",
        f"- {r['excluded_text_mismatch']} rows flagged `category_text_mismatch`",
        f"- {r['excluded_no_open_date']} rows without a valid open date",
        f"- {r['not_evaluable']} rows whose previous same-category fault was closed with an "
        "unknown close date (cannot be judged)",
        "",
        "### How sensitive is the repeat rate to the window?",
        "",
        "If the rate swings widely with the window, it is measuring how busy a system is rather "
        "than whether faults come back, so the window is a choice worth seeing.",
        "",
        md_table(_show(_sens_table(m.repeat_sensitivity, r["window_days"]))),
        "",
        "### Top 10 repeat-offender systems (by repeat rate)",
        "",
        md_table(_show(_rate_table(m.top_offenders))),
        "",
        "### Repeat rate by vendor",
        "",
        md_table(_show(_rate_table(m.repeat_by_vendor))),
        "",
        "## Resolution time (MTTR)",
        "",
        "Only properly closed tickets: status Closed with a usable close date. `tickets` is all "
        "tickets in the group, `n` the resolved ones behind the numbers.",
        "",
    ]
    for name, table in m.mttr.items():
        if name == "system":
            table = table.head(12)
            out += [f"### By {name} (12 slowest)", ""]
        else:
            out += [f"### By {name}", ""]
        out += [md_table(_show(table)), ""]
    out += [
        "## Open backlog and aging",
        "",
        "Open = status Open or In Progress. Age = days from the open date to the reporting date.",
        "",
        md_table(_show(m.aging_overall)),
        "",
        "### By site",
        "",
        md_table(_show(m.aging_by_site)),
        "",
        "### By vendor",
        "",
        md_table(_show(m.aging_by_vendor)),
        "",
        "## Monthly trend",
        "",
        "`median_days` covers tickets closed that month. `open_at_month_end` leaves out tickets "
        "closed with an unknown close date.",
        "",
        md_table(_show(m.monthly)),
        "",
    ]
    if checks is not None:
        out += [
            "## Did the metrics surface the planted patterns? (synthetic data only)",
            "",
            "Criteria were fixed before looking at the results. A pattern that is not "
            "surfaced is a finding about the metric, not a reason to change the generator.",
            "",
            md_table(checks),
            "",
        ]
    return "\n".join(out)
