"""Standardize raw fault reports field by field, validate them and remove duplicates.

Rules this module follows:
- Never guess. A value that cannot be mapped with confidence becomes "Unknown" (or a missing
  date) and the reason is recorded, so the data-quality report can explain every Unknown.
- Vendors are matched by pattern, never by similarity: "Vendor A" and "Vendor B" differ by one
  character, so a similarity score cannot tell a typo from a different vendor.
- Fuzzy matching is used for categories and sites only (spelling typos in long, distinct words).
- This module never reads the ground-truth files; scoring lives in ``evaluate.py``.
"""

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cache

import pandas as pd
from rapidfuzz import fuzz, process

from field_ops import config
from field_ops.aliases import CATEGORY_ALIASES, SEVERITY_ALIASES, STATUS_ALIASES
from field_ops.dedupe import (
    DEDUPE_SIMILARITY,
    choose_survivor,
    description_key,
    near_duplicate_clusters,
)

UNKNOWN = "Unknown"
FUZZY_CUTOFF = 80  # lowest similarity (0-100) accepted as a typo match
FUZZY_CONFIDENT = 85  # at or above this a match is trusted; 80-85 is flagged for review
FUZZY_MARGIN = 5  # best match must beat the runner-up by this much, else it is ambiguous

RAW_COLUMNS = [
    "report_id", "date_opened", "date_closed", "site", "system_type", "system_id",
    "vendor", "fault_category", "severity", "description", "reported_by", "status",
]  # fmt: skip

CLEAN_COLUMNS = [*RAW_COLUMNS, "dq_flags", "dup_count", "merged_report_ids"]

_FIELD_METHOD = {
    "vendor": "m_vendor",
    "fault_category": "m_category",
    "site": "m_site",
    "system_id": "m_system_id",
    "system_type": "m_system_type",
    "severity": "m_severity",
    "status": "m_status",
    "reported_by": "m_reporter",
}


# ---------------------------------------------------------------------------------------------
# Ledger: every fix is recorded as (step, rule, rows affected, example)
# ---------------------------------------------------------------------------------------------
@dataclass
class Ledger:
    entries: list[dict] = field(default_factory=list)

    def add(self, step: str, rule: str, rows: int, example: str = "") -> None:
        if rows:
            self.entries.append({"step": step, "rule": rule, "rows": int(rows), "example": example})

    def add_changes(self, step: str, rule: str, before: pd.Series, after: pd.Series) -> None:
        """Record how many rows differ between before/after, plus the commonest change."""
        b, a = before.fillna("").astype(str), after.fillna("").astype(str)
        changed = b != a
        if not changed.any():
            return
        pairs = pd.Series(list(zip(b[changed], a[changed], strict=True)))
        old, new = pairs.value_counts().index[0]
        self.add(step, rule, changed.sum(), f"'{old}' -> '{new}'")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries, columns=["step", "rule", "rows", "example"])


@dataclass
class CleanResult:
    clean: pd.DataFrame  # one row per real ticket, output columns only
    rows: pd.DataFrame  # every raw row after standardization, with methods and cluster ids
    survivor_index: pd.Index  # index labels in ``rows`` of the kept rows, in ``clean`` order
    ledger: Ledger
    assumptions: list[str]


# ---------------------------------------------------------------------------------------------
# Scalar normalizers. Each returns (clean value, method) and is cached because the number of
# distinct raw values is tiny compared with the number of rows.
# ---------------------------------------------------------------------------------------------
def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


_VENDOR_RE = re.compile(r"^(?:vendor|vndr|vendr)[\s\-_]*([a-z])(?:\s*(?:ltd|inc)\.?)?$")
_VENDOR_LETTERS = {v.split()[-1].lower() for v in config.VENDORS}


@cache
def clean_vendor(raw: str) -> tuple[str, str]:
    """Normalize, extract the letter, rebuild "Vendor X". No similarity matching."""
    key = _squash(raw).lower()
    if not key:
        return UNKNOWN, "unknown_blank"
    match = _VENDOR_RE.match(key)
    if not match:
        return UNKNOWN, "unknown_unparseable"
    letter = match.group(1)
    if letter not in _VENDOR_LETTERS:
        return UNKNOWN, "unknown_not_a_vendor"
    canonical = f"Vendor {letter.upper()}"
    return canonical, "exact" if raw == canonical else "pattern"


def _fuzzy(key: str, choices: dict[str, str]) -> tuple[str | None, str]:
    hits = process.extract(
        key, list(choices), scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF, limit=2
    )
    if not hits:
        return None, "unknown_no_match"
    if len(hits) == 2 and hits[0][1] - hits[1][1] < FUZZY_MARGIN:
        return None, "unknown_ambiguous"
    method = "fuzzy" if hits[0][1] >= FUZZY_CONFIDENT else "fuzzy_low"
    return choices[hits[0][0]], method


def _match_label(
    raw: str,
    canonical: list[str],
    aliases: dict[str, str],
    *,
    fuzzy: bool,
    strip_suffix: str = "",
) -> tuple[str, str]:
    """Resolve a label: canonical spelling, then alias table, then (optionally) fuzzy match."""
    key = _squash(raw).lower()
    if strip_suffix:
        key = re.sub(rf"\s+{strip_suffix}$", "", key)
    if not key:
        return UNKNOWN, "unknown_blank"
    by_key = {c.lower(): c for c in canonical}
    if key in by_key:
        return by_key[key], "exact" if raw == by_key[key] else "pattern"
    if key in aliases:
        return aliases[key], "alias"
    if fuzzy:
        value, method = _fuzzy(key, by_key)
        return (value or UNKNOWN), method
    return UNKNOWN, "unknown_no_match"


@cache
def clean_category(raw: str) -> tuple[str, str]:
    return _match_label(raw, config.CATEGORIES, CATEGORY_ALIASES, fuzzy=True)


@cache
def clean_site(raw: str) -> tuple[str, str]:
    return _match_label(raw, config.SITES, {}, fuzzy=True, strip_suffix="site")


@cache
def clean_severity(raw: str) -> tuple[str, str]:
    return _match_label(raw, config.SEVERITIES, SEVERITY_ALIASES, fuzzy=False)


@cache
def clean_status(raw: str) -> tuple[str, str]:
    return _match_label(raw, config.STATUSES, STATUS_ALIASES, fuzzy=False)


_SYSTEM_ID_RE = re.compile(r"^([a-z]{3})[\s\-]?0*(\d{1,2})$")


@cache
def clean_system_id(raw: str) -> tuple[str, str]:
    key = _squash(raw).lower()
    if not key:
        return UNKNOWN, "unknown_blank"
    match = _SYSTEM_ID_RE.match(key)
    if not match or match.group(1).upper() not in config.SYSTEM_TYPES:
        return UNKNOWN, "unknown_unparseable"
    canonical = f"{match.group(1).upper()}-{int(match.group(2)):02d}"
    return canonical, "exact" if raw == canonical else "pattern"


def derive_system_type(system_id: str, raw_type: str) -> tuple[str, str]:
    """The ID prefix is the reliable key; the free-text type is only a fallback."""
    prefix = system_id.split("-")[0]
    if prefix in config.SYSTEM_TYPES:
        return config.SYSTEM_TYPES[prefix], "derived_from_id"
    by_key = {t.lower(): t for t in config.SYSTEM_TYPES.values()}
    key = _squash(raw_type).lower()
    if key in by_key:
        return by_key[key], "pattern"
    return UNKNOWN, "unknown_no_match"


def clean_description(raw: str) -> str:
    text = _squash(raw)
    text = re.sub(r"\.{2,}$", ".", text)
    return text.capitalize() if text.isupper() else text


_INITIAL_RE = re.compile(r"^[a-z]\.?$", re.IGNORECASE)


def normalize_name(raw: str) -> str:
    """ "last, first" -> "First Last"; fix case. Initials are expanded later from a roster."""
    text = _squash(raw)
    if "," in text:
        last, _, first = text.partition(",")
        text = f"{first.strip()} {last.strip()}"
    return text.title()


def build_roster(names: pd.Series) -> dict[tuple[str, str], set[str]]:
    """Full names seen in the data, keyed by (first initial, last name)."""
    roster: dict[tuple[str, str], set[str]] = {}
    for name in names.unique():
        parts = name.split(" ", 1)
        if len(parts) == 2 and not _INITIAL_RE.match(parts[0]):
            roster.setdefault((parts[0][0].lower(), parts[1].lower()), set()).add(name)
    return roster


def clean_reporter(raw: str, roster: dict[tuple[str, str], set[str]]) -> tuple[str, str]:
    name = normalize_name(raw)
    if not name:
        return UNKNOWN, "unknown_blank"
    parts = name.split(" ", 1)
    if len(parts) == 2 and _INITIAL_RE.match(parts[0]):
        candidates = roster.get((parts[0][0].lower(), parts[1].lower()), set())
        if len(candidates) == 1:
            return next(iter(candidates)), "initial_expanded"
        return name, "partial_name"
    return name, "exact" if raw == name else "pattern"


# ---------------------------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------------------------
_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$")
_DMY_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_TEXT_RE = re.compile(r"^([a-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})$")
_SERIAL_RE = re.compile(r"^\d{5}(?:\.\d+)?$")
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
)}  # fmt: skip


@cache
def parse_date(raw: str) -> tuple[date | None, str]:
    """Parse one of the known formats. Slash dates are DD/MM/YYYY (documented assumption)."""
    text = raw.strip()
    if not text:
        return None, "blank"
    try:
        if m := _ISO_RE.match(text):
            return date(int(m[1]), int(m[2]), int(m[3])), "iso"
        if m := _DMY_RE.match(text):
            day, month = int(m[1]), int(m[2])
            ambiguous = day <= 12 and month <= 12 and day != month
            return date(int(m[3]), month, day), "dmy_ambiguous" if ambiguous else "dmy"
        if (m := _TEXT_RE.match(text.lower())) and m[1] in _MONTHS:
            return date(int(m[3]), _MONTHS[m[1]], int(m[2])), "text"
        if _SERIAL_RE.match(text):
            return config.EXCEL_EPOCH + timedelta(days=int(float(text))), "excel_serial"
    except (ValueError, OverflowError):
        pass
    return None, "unparseable"


def _apply(series: pd.Series, fn) -> tuple[pd.Series, pd.Series]:
    pairs = series.map(fn)
    return pairs.map(lambda p: p[0]), pairs.map(lambda p: p[1])


def _parse_dates(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    values, methods = _apply(series, parse_date)
    return pd.to_datetime(values), methods


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------
def _standardize(raw: pd.DataFrame, ledger: Ledger) -> pd.DataFrame:
    rows = pd.DataFrame({"report_id": raw["report_id"].map(_squash).str.upper()})
    step = "standardize"
    ledger.add_changes(
        step, "report_id: trimmed and upper-cased", raw["report_id"], rows["report_id"]
    )

    rows["vendor"], rows["m_vendor"] = _apply(raw["vendor"], clean_vendor)
    ledger.add_changes(
        step, "vendor: normalized to 'Vendor X' (pattern, no fuzzy)", raw["vendor"], rows["vendor"]
    )

    rows["fault_category"], rows["m_category"] = _apply(raw["fault_category"], clean_category)
    for method, rule in [
        ("alias", "fault_category: mapped through alias table"),
        ("fuzzy", "fault_category: spelling typo fixed by fuzzy match"),
        ("fuzzy_low", "fault_category: typo fixed by low-confidence fuzzy match (80-85)"),
        ("pattern", "fault_category: case/whitespace normalized"),
    ]:
        mask = rows["m_category"] == method
        ledger.add_changes(step, rule, raw["fault_category"][mask], rows["fault_category"][mask])

    rows["site"], rows["m_site"] = _apply(raw["site"], clean_site)
    ledger.add_changes(step, "site: case/whitespace/typo normalized", raw["site"], rows["site"])

    rows["system_id"], rows["m_system_id"] = _apply(raw["system_id"], clean_system_id)
    ledger.add_changes(
        step, "system_id: normalized to 'XXX-NN'", raw["system_id"], rows["system_id"]
    )

    derived = [
        derive_system_type(i, t) for i, t in zip(rows["system_id"], raw["system_type"], strict=True)
    ]
    rows["system_type"] = [d[0] for d in derived]
    rows["m_system_type"] = [d[1] for d in derived]
    ledger.add_changes(
        step, "system_type: derived from system_id prefix", raw["system_type"], rows["system_type"]
    )

    rows["severity"], rows["m_severity"] = _apply(raw["severity"], clean_severity)
    ledger.add_changes(
        step, "severity: mapped to Low/Medium/High/Critical", raw["severity"], rows["severity"]
    )

    rows["status"], rows["m_status"] = _apply(raw["status"], clean_status)
    ledger.add_changes(
        step, "status: mapped to Open/In Progress/Closed", raw["status"], rows["status"]
    )

    names = raw["reported_by"].map(normalize_name)
    roster = build_roster(names)
    reporters = [clean_reporter(r, roster) for r in raw["reported_by"]]
    rows["reported_by"] = [r[0] for r in reporters]
    rows["m_reporter"] = [r[1] for r in reporters]
    ledger.add_changes(
        step,
        "reported_by: name format and initials resolved",
        raw["reported_by"],
        rows["reported_by"],
    )

    rows["description"] = raw["description"].map(clean_description)
    ledger.add_changes(
        step, "description: whitespace/shouting normalized", raw["description"], rows["description"]
    )
    rows["description_key"] = rows["description"].map(description_key)
    return rows


def _dates_and_validation(
    raw: pd.DataFrame,
    rows: pd.DataFrame,
    ledger: Ledger,
    min_date: date,
    max_date: date,
) -> None:
    rows["date_opened"], rows["m_date_opened"] = _parse_dates(raw["date_opened"])
    rows["date_closed"], rows["m_date_closed"] = _parse_dates(raw["date_closed"])
    lo, hi = pd.Timestamp(min_date), pd.Timestamp(max_date)

    for col in ("date_opened", "date_closed"):
        out_of_range = rows[col].notna() & ((rows[col] < lo) | (rows[col] > hi))
        rows.loc[out_of_range, f"m_{col}"] = "out_of_range"
        rows.loc[out_of_range, col] = pd.NaT
        iso = rows[col].dt.strftime("%Y-%m-%d")
        ledger.add_changes("standardize", f"{col}: converted to ISO date", raw[col], iso.fillna(""))
        bad = rows[f"m_{col}"].isin(["unparseable", "out_of_range"])
        ledger.add(
            "validate", f"{col}: unparseable or outside plausible range -> missing", bad.sum()
        )

    before = rows["date_closed"].copy()
    swapped = (
        rows["date_closed"].notna()
        & rows["date_opened"].notna()
        & (rows["date_closed"] < rows["date_opened"])
    )
    rows.loc[swapped, "date_closed"] = pd.NaT
    example = f"{before[swapped].dt.date.iloc[0]} -> missing" if swapped.any() else ""
    ledger.add(
        "validate", "date_closed before date_opened -> close date removed", swapped.sum(), example
    )

    reopened = rows["status"].isin(["Open", "In Progress"]) & rows["date_closed"].notna()
    ledger.add(
        "validate",
        "status Open/In Progress but has close date -> Closed",
        reopened.sum(),
        "'Open' -> 'Closed'",
    )
    rows.loc[reopened, "status"] = "Closed"

    rows["flag_close_before_open"] = swapped
    rows["flag_status_closed_from_date"] = reopened
    rows["flag_closed_no_close_date"] = (rows["status"] == "Closed") & rows["date_closed"].isna()
    ledger.add(
        "validate",
        "status Closed but no usable close date -> kept, excluded from MTTR",
        rows["flag_closed_no_close_date"].sum(),
    )


def _dq_flags(rows: pd.DataFrame) -> pd.Series:
    flags = {
        "close_before_open": rows["flag_close_before_open"],
        "status_closed_from_date": rows["flag_status_closed_from_date"],
        "closed_no_close_date": rows["flag_closed_no_close_date"],
        "date_invalid": rows["m_date_opened"].isin(["unparseable", "out_of_range"])
        | rows["m_date_closed"].isin(["unparseable", "out_of_range"]),
        "category_unknown": rows["fault_category"] == UNKNOWN,
        "fuzzy_low_confidence": (rows["m_category"] == "fuzzy_low")
        | (rows["m_site"] == "fuzzy_low"),
        "vendor_unknown": rows["vendor"] == UNKNOWN,
        "duplicate_merged": rows["dup_count"] > 1,
    }
    out = pd.Series("", index=rows.index)
    for name, mask in flags.items():
        out = out.where(~mask, out + name + ";")
    return out.str.rstrip(";")


def _report_id_number(report_id: str) -> int:
    digits = re.sub(r"\D", "", report_id)
    return int(digits) if digits else 0


def clean_reports(
    raw: pd.DataFrame,
    *,
    min_date: date | None = None,
    max_date: date = config.AS_OF,
    similarity: int = DEDUPE_SIMILARITY,
) -> CleanResult:
    """Run the full cleaning pipeline on the raw export (all columns as strings)."""
    raw = raw[RAW_COLUMNS].fillna("").astype(str).reset_index(drop=True)
    min_date = min_date or (config.START_DATE - timedelta(days=366))
    ledger = Ledger()

    rows = _standardize(raw, ledger)
    _dates_and_validation(raw, rows, ledger, min_date, max_date)

    # Stage 1: byte-identical exports of the same row.
    group = raw.groupby(RAW_COLUMNS, sort=False).ngroup()
    rep_of_row = pd.Series(raw.index, index=raw.index).groupby(group).transform("first")
    is_rep = rep_of_row == raw.index
    ledger.add(
        "dedupe", "exact duplicate rows removed", (~is_rep).sum(), "identical row exported twice"
    )

    # Stage 2: same fault entered again as a new ticket (system + day + category + similar text).
    reps = rows[is_rep]
    labels = near_duplicate_clusters(reps, similarity)
    rows["cluster_id"] = rep_of_row.map(labels)
    near_removed = len(reps) - labels.nunique()
    ledger.add(
        "dedupe",
        f"near-duplicate tickets merged (same system/day/category, text >= {similarity}% similar)",
        near_removed,
        "kept: closed-with-date, then most complete, then earliest",
    )

    survivors = {}
    for cluster_id, members in reps.groupby(rows.loc[is_rep, "cluster_id"]):
        survivors[cluster_id] = choose_survivor(members)
    rows["is_survivor"] = rows.index.isin(list(survivors.values()))
    rows["dup_count"] = rows.groupby("cluster_id")["report_id"].transform("size")
    ids = rows.groupby("cluster_id")["report_id"].agg(lambda s: ";".join(sorted(set(s))))
    rows["merged_report_ids"] = rows["cluster_id"].map(ids)
    rows["dq_flags"] = _dq_flags(rows)

    clean = rows[rows["is_survivor"]].copy()
    clean["_n"] = clean["report_id"].map(_report_id_number)
    clean = clean.sort_values("_n", kind="stable")
    for col in ("date_opened", "date_closed"):
        clean[col] = clean[col].dt.strftime("%Y-%m-%d").fillna("")
    survivor_index = clean.index
    clean = clean[CLEAN_COLUMNS].reset_index(drop=True)

    ambiguous = int(
        (
            rows["m_date_opened"].eq("dmy_ambiguous") | rows["m_date_closed"].eq("dmy_ambiguous")
        ).sum()
    )
    assumptions = [
        "Slash dates are read as DD/MM/YYYY (source-system locale). "
        f"{ambiguous} rows contain a slash date where DD/MM and MM/DD both look valid.",
        "Excel serial dates use the 1899-12-30 epoch.",
        "Vendors are matched by pattern only; anything that is not 'Vendor <A-O>' becomes Unknown.",
        f"Fuzzy matching (categories and sites only) accepts similarity >= {FUZZY_CUTOFF}, with a "
        f"runner-up margin of {FUZZY_MARGIN}; matches scoring {FUZZY_CUTOFF}-{FUZZY_CONFIDENT} "
        "are flagged fuzzy_low_confidence for review.",
        "Alias tables (aliases.py) come from profiling raw values; plain typos are not listed.",
        "system_type is derived from the system_id prefix; the free-text type is only a fallback.",
        "Descriptions are tidied (whitespace, shouting) but not spell-corrected.",
        "Blank categories stay Unknown here; the classification step fills them from the text.",
    ]
    return CleanResult(
        clean=clean,
        rows=rows,
        survivor_index=survivor_index,
        ledger=ledger,
        assumptions=assumptions,
    )
