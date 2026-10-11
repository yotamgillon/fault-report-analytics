"""Turn clean truth tickets into the messy raw export a field team actually produces.

Every kind of mess is applied at a known rate and counted, so the cleaning step can later
be scored against what was really injected. The raw output has only the 12 business
columns; anything that reveals the answers stays in the truth files.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from field_ops.config import EXCEL_EPOCH

RAW_COLUMNS = [
    "report_id", "date_opened", "date_closed", "site", "system_type", "system_id",
    "vendor", "fault_category", "severity", "description", "reported_by", "status",
]  # fmt: skip

TRUTH_COLUMNS = [
    "report_id", "date_opened", "date_closed", "site", "system_type", "system_id", "vendor",
    "fault_category", "logged_category", "description_type", "secondary_category", "severity",
    "description", "reported_by", "status", "is_lemon_system", "is_recurrence",
    "parent_report_id", "long_tail", "stuck",
]  # fmt: skip

MESS_RATES = {
    "exact_duplicate": 0.04,  # same row exported twice
    "near_duplicate": 0.03,  # same fault re-entered as a new ticket
    "near_duplicate_stale_status": 0.40,  # share of near-dups still showing "Open"
    "vendor_variant": 0.30,
    "category_blank": 0.08,
    "category_variant": 0.25,
    "site_variant": 0.15,
    "system_type_variant": 0.15,
    "system_id_variant": 0.15,
    "severity_variant": 0.15,
    "status_variant": 0.20,
    "reporter_variant": 0.12,
    "description_typo": 0.15,
    "description_upper": 0.04,
    "description_whitespace": 0.08,
    "missing_close_date": 0.10,  # of closed tickets
    "close_before_open": 0.015,  # of closed tickets
    "open_with_close_date": 0.02,  # of closed tickets: status never updated
}

# Date formats seen in the export, with their share. Slash dates are always DD/MM/YYYY
# (the source system's locale) - the cleaner must document that assumption.
DATE_FORMATS = {"iso": 0.50, "dmy": 0.25, "text": 0.12, "excel_serial": 0.08, "iso_time": 0.05}

CATEGORY_VARIANTS = {
    "Power": ["power", "POWER", "Pwr", "Power Supply", "Battery/Power", "Powr"],
    "Communications": ["Comms", "comms", "Communication", "Comm", "COMMUNICATIONS",
                       "Comunications", "Network"],
    "Calibration": ["Cal", "calibration", "Cal/Align", "Calibraton", "CAL"],
    "Sensor": ["Sensors", "sensor", "Camera/Sensor", "Detector", "Snsor"],
    "Software": ["SW", "software", "Softwear", "S/W", "Software Bug"],
    "Mechanical": ["Mech", "mechanical", "Mechanic", "Frame/Mech", "Mechancial"],
}  # fmt: skip

SYSTEM_TYPE_VARIANTS = {
    "Control Unit": ["CTL", "Ctrl Unit", "control unit", "Control Unt"],
    "Comms Module": ["Comm Module", "comms module", "COMMS MODULE", "Communications Module"],
    "Sensor Module": ["sensor module", "Sensor-Module", "SensorModule", "Sensr Module"],
    "Power Unit": ["power unit", "PWR Unit", "Power-Unit", "Powerunit"],
    "Mechanical Frame": ["mechanical frame", "MECHANICAL FRAME", "Mech Frame", "Mechanical Frme"],
}  # fmt: skip

SEVERITY_VARIANTS = {
    "Low": ["low", "LOW", "Lo", "L"],
    "Medium": ["medium", "Med", "MED", "Moderate"],
    "High": ["high", "HIGH", "Hi", "Hgh"],
    "Critical": ["critical", "Crit", "CRITICAL", "Critcal"],
}

STATUS_VARIANTS = {
    "Open": ["open", "OPEN", "New"],
    "In Progress": ["in progress", "In-Progress", "WIP", "in prog"],
    "Closed": ["closed", "CLOSED", "Resolved", "Done"],
}


@dataclass
class MessyDataset:
    raw: pd.DataFrame
    truth: pd.DataFrame
    duplicates: pd.DataFrame
    summary: dict = field(default_factory=dict)


def _pick(rng: np.random.Generator, items):
    return items[int(rng.integers(len(items)))]


def typo(rng: np.random.Generator, text: str) -> str:
    """Swap, drop or double one character inside a random word of 4+ letters."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not candidates:
        return text
    i = _pick(rng, candidates)
    w = words[i]
    pos = int(rng.integers(1, len(w) - 1))
    op = int(rng.integers(3))
    if op == 0:
        w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2 :]
    elif op == 1:
        w = w[:pos] + w[pos + 1 :]
    else:
        w = w[:pos] + w[pos] + w[pos:]
    words[i] = w
    return " ".join(words)


def format_date(rng: np.random.Generator, d: date, style: str) -> str:
    if style == "iso":
        return d.isoformat()
    if style == "dmy":
        return d.strftime("%d/%m/%Y")
    if style == "text":
        return f"{d:%b} {d.day} {d.year}"
    if style == "excel_serial":
        return str((d - EXCEL_EPOCH).days)
    return f"{d.isoformat()} {int(rng.integers(6, 23)):02d}:{int(rng.integers(60)):02d}"


def vendor_variant(rng: np.random.Generator, vendor: str) -> str:
    letter = vendor.split()[-1]
    return _pick(
        rng,
        [
            f"vendor {letter.lower()}", f"VENDOR {letter}", f"Vendor-{letter}", f"Vndr {letter}",
            f"Vendor{letter}", f"Vendor {letter} ", f" vendor {letter}", f"Vendor  {letter}",
            f"Vendor {letter} Ltd", f"Vendr {letter}",
        ],
    )  # fmt: skip


def site_variant(rng: np.random.Generator, site: str) -> str:
    return _pick(rng, [site.lower(), site.upper(), f" {site}", f"{site} Site", typo(rng, site)])


def system_id_variant(rng: np.random.Generator, system_id: str) -> str:
    prefix, num = system_id.split("-")
    return _pick(
        rng,
        [system_id.lower(), f"{prefix}{num}", f"{prefix}-{int(num)}", f"{prefix} {num}"],
    )


def reporter_variant(rng: np.random.Generator, name: str) -> str:
    first, last = name.split(" ", 1)
    return _pick(rng, [name.lower(), name.upper(), f"{first[0]}. {last}", f"{last}, {first}"])


def _light_edit(rng: np.random.Generator, text: str) -> str:
    """A re-entered ticket: same fault, slightly different wording."""
    op = int(rng.integers(4))
    if op == 0:
        return typo(rng, text)
    if op == 1:
        return text.lower()
    if op == 2:
        return text.rstrip(".")
    return text.replace(" ", "  ", 1)


def _add_near_duplicates(truth: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = round(len(truth) * MESS_RATES["near_duplicate"])
    near = truth.loc[rng.choice(truth.index, size=n, replace=False)].copy()
    near["dup_of"] = near["ticket_key"]
    near["entry_ts"] = near["entry_ts"] + rng.uniform(0.3, 4.0, size=n)
    near["description"] = [_light_edit(rng, d) for d in near["description"]]
    near["reported_by"] = [
        _pick(rng, [r for r in near["reported_by"] if r != rep]) if rng.random() < 0.5 else rep
        for rep in near["reported_by"]
    ]
    stale = (near["status"] == "Closed") & (
        rng.random(n) < MESS_RATES["near_duplicate_stale_status"]
    )
    near.loc[stale, "status"] = "Open"
    near.loc[stale, "date_closed"] = None
    return near


def _corrupt_row(rng: np.random.Generator, row: dict, counts: dict) -> dict:
    def hit(kind: str) -> bool:
        if rng.random() < MESS_RATES[kind]:
            counts[kind] = counts.get(kind, 0) + 1
            return True
        return False

    out = {"report_id": row["report_id"]}

    styles, weights = zip(*DATE_FORMATS.items(), strict=True)
    open_style = styles[rng.choice(len(styles), p=weights)]
    close_style = open_style if rng.random() < 0.8 else styles[rng.choice(len(styles), p=weights)]
    counts[f"date_format_{open_style}"] = counts.get(f"date_format_{open_style}", 0) + 1

    opened: date = row["date_opened"]
    closed: date | None = row["date_closed"]
    status = row["status"]
    if status == "Closed" and closed is not None:
        r = rng.random()
        missing = MESS_RATES["missing_close_date"]
        before = missing + MESS_RATES["close_before_open"]
        if r < missing:
            closed = None
            counts["missing_close_date"] = counts.get("missing_close_date", 0) + 1
        elif r < before:
            closed = opened - timedelta(days=int(rng.integers(1, 16)))
            counts["close_before_open"] = counts.get("close_before_open", 0) + 1
        elif r < before + MESS_RATES["open_with_close_date"]:
            status = "Open"
            counts["open_with_close_date"] = counts.get("open_with_close_date", 0) + 1

    out["date_opened"] = format_date(rng, opened, open_style)
    out["date_closed"] = format_date(rng, closed, close_style) if closed else ""

    out["site"] = site_variant(rng, row["site"]) if hit("site_variant") else row["site"]
    out["system_type"] = (
        _pick(rng, SYSTEM_TYPE_VARIANTS[row["system_type"]])
        if hit("system_type_variant")
        else row["system_type"]
    )
    out["system_id"] = (
        system_id_variant(rng, row["system_id"]) if hit("system_id_variant") else row["system_id"]
    )
    out["vendor"] = vendor_variant(rng, row["vendor"]) if hit("vendor_variant") else row["vendor"]

    category = row["logged_category"]
    if hit("category_blank"):
        category = ""
    elif hit("category_variant"):
        category = _pick(rng, CATEGORY_VARIANTS[category])
    out["fault_category"] = category

    out["severity"] = (
        _pick(rng, SEVERITY_VARIANTS[row["severity"]])
        if hit("severity_variant")
        else row["severity"]
    )

    text = row["description"]
    if hit("description_typo"):
        text = typo(rng, text)
        if rng.random() < 0.3:
            text = typo(rng, text)
    if hit("description_upper"):
        text = text.upper()
    if hit("description_whitespace"):
        text = _pick(rng, [f"  {text}", f"{text}  ", text.replace(" ", "  "), f"{text}.."])
    out["description"] = text

    out["reported_by"] = (
        reporter_variant(rng, row["reported_by"]) if hit("reporter_variant") else row["reported_by"]
    )
    out["status"] = _pick(rng, STATUS_VARIANTS[status]) if hit("status_variant") else status
    return out


def make_messy(truth: pd.DataFrame, rng: np.random.Generator) -> MessyDataset:
    """Inject duplicates and field-level mess. Returns raw rows plus the answer key."""
    near = _add_near_duplicates(truth, rng)
    combined = pd.concat([truth.assign(dup_of=None), near], ignore_index=True)

    # Report IDs are sequential in order of entry, so a re-entered ticket gets a later ID.
    combined = combined.sort_values("entry_ts", kind="stable").reset_index(drop=True)
    combined["report_id"] = [f"FR-{10001 + i}" for i in range(len(combined))]
    key_to_id = dict(
        zip(
            combined.loc[combined["dup_of"].isna(), "ticket_key"],
            combined.loc[combined["dup_of"].isna(), "report_id"],
            strict=True,
        )
    )

    counts: dict[str, int] = {}
    raw = pd.DataFrame(
        [_corrupt_row(rng, row, counts) for row in combined.to_dict("records")],
        columns=RAW_COLUMNS,
    )

    # Exact duplicates: whole rows exported twice, ending up anywhere in the file.
    n_exact = round(len(raw) * MESS_RATES["exact_duplicate"])
    exact = raw.loc[rng.choice(raw.index, size=n_exact, replace=False)]
    order = np.concatenate([np.arange(len(raw)), rng.uniform(0, len(raw), size=n_exact)])
    raw = pd.concat([raw, exact], ignore_index=True).iloc[np.argsort(order, kind="stable")]
    raw = raw.reset_index(drop=True)
    counts["exact_duplicate"] = n_exact
    counts["near_duplicate"] = len(near)

    is_near = combined["dup_of"].notna()
    near_original = dict(
        zip(
            combined.loc[is_near, "report_id"],
            combined.loc[is_near, "dup_of"].map(key_to_id),
            strict=True,
        )
    )
    duplicates = pd.concat(
        [
            pd.DataFrame(
                {
                    "report_id": combined.loc[is_near, "report_id"],
                    "original_report_id": combined.loc[is_near, "dup_of"].map(key_to_id),
                    "kind": "near",
                }
            ),
            pd.DataFrame(
                {
                    "report_id": exact["report_id"],
                    # A row exported twice may itself be a re-entered ticket: point at the original.
                    "original_report_id": exact["report_id"]
                    .map(near_original)
                    .fillna(exact["report_id"]),
                    "kind": "exact",
                }
            ),
        ],
        ignore_index=True,
    )

    truth_out = combined.loc[~is_near].copy()
    truth_out["parent_report_id"] = truth_out["parent_key"].map(key_to_id)
    truth_out = truth_out[TRUTH_COLUMNS].sort_values("report_id").reset_index(drop=True)

    summary = {
        "raw_rows": len(raw),
        "unique_tickets": len(truth_out),
        "date_assumption": "Slash dates are DD/MM/YYYY; Excel serials use the 1899-12-30 epoch.",
        "mess_rates": MESS_RATES,
        "date_formats": DATE_FORMATS,
        "injected_counts": dict(sorted(counts.items())),
        "description_types": truth_out["description_type"].value_counts().to_dict(),
    }
    return MessyDataset(raw=raw, truth=truth_out, duplicates=duplicates, summary=summary)
