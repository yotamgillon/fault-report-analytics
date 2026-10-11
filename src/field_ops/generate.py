"""Simulate a year of field fault reports with planted, detectable patterns.

This module produces the *clean truth*: one row per real ticket. ``mess.py`` then turns it
into the messy raw CSV the pipeline sees. Everything is synthetic and generic.

Planted patterns (subtle on purpose, found only through analysis):
- three vendors resolve faults more slowly (MTTR multiplier 1.3-1.45, inside heavy noise)
- one vendor has a few very long "waiting for parts" tickets: bad mean, normal median
- five "lemon" systems fail more often and have more repeat faults
- one site leaves more tickets stuck open, so its backlog ages
- summer brings more reports, skewed toward power and sensor faults
"""

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from field_ops import config
from field_ops.mess import MessyDataset, make_messy


@dataclass(frozen=True)
class Scenario:
    """Settings for the synthetic world. The defaults are what the README insights describe."""

    seed: int = config.SEED
    n_base_faults: int = 1760
    lemon_systems: tuple[str, ...] = ("CTL-03", "COM-04", "SEN-02", "PWR-03", "FRM-05")
    lemon_weight: float = 1.3
    recurrence_prob: float = 0.06
    lemon_recurrence_prob: float = 0.20
    slow_vendors: tuple[tuple[str, float], ...] = (
        ("Vendor F", 1.45),
        ("Vendor K", 1.35),
        ("Vendor N", 1.30),
    )
    long_tail_vendor: str = "Vendor C"
    long_tail_prob: float = 0.05
    backlog_site: str = "Ridgeview"
    backlog_slowdown: float = 1.2
    stuck_prob: float = 0.025
    backlog_stuck_prob: float = 0.08
    vague_rate: float = 0.08
    multi_rate: float = 0.07
    mismatch_rate: float = 0.05


MONTH_WEIGHT = {
    10: 1.0, 11: 0.95, 12: 0.85, 1: 0.9, 2: 0.9, 3: 1.0,
    4: 1.0, 5: 1.05, 6: 1.2, 7: 1.3, 8: 1.25, 9: 1.05,
}  # fmt: skip
SUMMER_MONTHS = {6, 7, 8}
WEEKEND_WEIGHT = 0.55

# Probability of each fault category per system type (same order as config.CATEGORIES:
# Power, Communications, Calibration, Sensor, Software, Mechanical).
CATEGORY_MIX = {
    "CTL": [0.15, 0.20, 0.10, 0.10, 0.35, 0.10],
    "COM": [0.15, 0.50, 0.05, 0.05, 0.15, 0.10],
    "SEN": [0.10, 0.10, 0.05, 0.50, 0.15, 0.10],
    "PWR": [0.55, 0.10, 0.05, 0.05, 0.10, 0.15],
    "FRM": [0.15, 0.10, 0.20, 0.05, 0.05, 0.45],
}
SUMMER_BOOST = {"Power": 1.5, "Sensor": 1.2}

SEVERITY_PROBS = [0.30, 0.40, 0.22, 0.08]
SEVERITY_MEDIAN_DAYS = {"Low": 8.0, "Medium": 6.0, "High": 4.0, "Critical": 2.5}
RESOLUTION_SIGMA = 0.7  # lognormal noise: big overlap between vendors

# Free-text building blocks: (subjects, symptoms) per category.
PHRASES = {
    "Power": (
        ["battery pack", "main battery", "power supply unit", "charging port",
         "voltage regulator", "power harness"],
        ["drops voltage under load", "fails to hold charge", "overheats while charging",
         "shows an undervoltage warning", "shuts down unexpectedly", "gives fluctuating readings"],
    ),
    "Communications": (
        ["network link", "radio module", "antenna connector", "telemetry link", "wireless bridge"],
        ["loses signal intermittently", "shows high packet loss", "fails to connect at startup",
         "drops the connection at long range", "has weak signal strength"],
    ),
    "Calibration": (
        ["calibration unit", "alignment module", "reference clock", "timing module"],
        ["drifts out of tolerance", "reports an offset error", "fails calibration",
         "loses alignment", "takes too long to settle"],
    ),
    "Sensor": (
        ["camera module", "thermal sensor", "infrared camera", "optical sensor", "zoom lens"],
        ["produces a blurry image", "drifts off target", "has no video feed",
         "shows a noisy thermal image", "fails its self-test"],
    ),
    "Software": (
        ["control software", "controller firmware", "scheduling app",
         "operator UI", "log server"],
        ["crashes on startup", "freezes during file upload", "shows error code E-{code}",
         "fails to install the update", "resets settings after reboot"],
    ),
    "Mechanical": (
        ["drive belt", "bearing housing", "motor mount", "frame panel", "servo", "cooling fan"],
        ["is cracked", "makes an abnormal noise", "is loose after transport",
         "shows excessive vibration", "is worn and needs replacement"],
    ),
}  # fmt: skip

# Causes from another category, for "two categories in one sentence" descriptions.
TRIGGERS = {
    "Power": ["a power fluctuation", "a battery swap"],
    "Communications": ["a radio reset", "a link dropout"],
    "Calibration": ["a clock resync", "a reference reset"],
    "Sensor": ["a camera restart", "a lens cleaning"],
    "Software": ["a firmware update", "a software reboot"],
    "Mechanical": ["a hard impact", "a belt replacement"],
}

VAGUE = [
    "Unit not working properly.",
    "Intermittent fault, see attached log.",
    "Issue during shift, needs a technician.",
    "System failure reported by operator.",
    "Same problem as last week.",
    "Not working again.",
    "Fault code on screen, unclear what it means.",
    "Something wrong after maintenance, please check.",
]

CONTEXTS = [
    "during pre-start check", "mid-operation", "after transport", "on startup",
    "during routine maintenance", "",
]  # fmt: skip
HEAT_CONTEXT = "in high ambient heat"


def _pick(rng: np.random.Generator, items):
    return items[int(rng.integers(len(items)))]


def _cap(text: str) -> str:
    """Capitalize the first letter only ("UI panel" stays "UI panel")."""
    return text[:1].upper() + text[1:]


def build_systems(scenario: Scenario) -> pd.DataFrame:
    """30 systems: 6 per type, 2 per vendor, one system of each type at every site but one."""
    rows = []
    for type_idx, (prefix, type_name) in enumerate(config.SYSTEM_TYPES.items()):
        type_vendors = config.VENDORS[3 * type_idx : 3 * type_idx + 3]
        for n in range(1, 7):
            system_id = f"{prefix}-{n:02d}"
            rows.append(
                {
                    "system_id": system_id,
                    "prefix": prefix,
                    "system_type": type_name,
                    "vendor": type_vendors[(n - 1) // 2],
                    "site": config.SITES[n - 1],
                    "is_lemon": system_id in scenario.lemon_systems,
                }
            )
    return pd.DataFrame(rows)


def _category_probs(prefix: str, month: int) -> np.ndarray:
    probs = np.array(CATEGORY_MIX[prefix], dtype=float)
    if month in SUMMER_MONTHS:
        for cat, boost in SUMMER_BOOST.items():
            probs[config.CATEGORIES.index(cat)] *= boost
    return probs / probs.sum()


def describe(
    rng: np.random.Generator,
    category: str,
    kind: str,
    secondary: str | None = None,
    summer: bool = False,
    recurrence: bool = False,
) -> str:
    """Build a free-text description. ``kind`` is clean, vague or multi (two categories)."""
    if kind == "vague":
        return _pick(rng, VAGUE)

    subjects, symptoms = PHRASES[category]
    subject = _pick(rng, subjects)
    symptom = _pick(rng, symptoms).format(code=int(rng.integers(100, 1000)))

    if kind == "multi":
        trigger = _pick(rng, TRIGGERS[secondary])
        if rng.random() < 0.5:
            text = f"{_cap(subject)} {symptom} after {trigger}."
        else:
            text = f"{_cap(subject)} {symptom}, possibly caused by {trigger}."
    else:
        context = _pick(rng, CONTEXTS)
        if summer and category in SUMMER_BOOST and rng.random() < 0.3:
            context = HEAT_CONTEXT
        style = int(rng.integers(3))
        if style == 0:
            text = f"{_cap(subject)} {symptom} {context}".strip() + "."
        elif style == 1 and context:
            text = f"{_cap(context)}: {subject} {symptom}."
        else:
            text = f"Operator reports {subject} {symptom}."

    if recurrence and rng.random() < 0.25:
        text = _pick(rng, ["Again: ", "Recurring - ", "Repeat issue: "]) + text
    return text


def simulate_truth(scenario: Scenario, rng: np.random.Generator) -> pd.DataFrame:
    """Return one clean row per real ticket, including hidden truth columns."""
    systems = build_systems(scenario)
    days = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    day_w = np.array(
        [MONTH_WEIGHT[d.month] * (WEEKEND_WEIGHT if d.dayofweek >= 5 else 1.0) for d in days]
    )
    sys_w = np.where(systems["is_lemon"], scenario.lemon_weight, 1.0)

    n = scenario.n_base_faults
    day_idx = rng.choice(len(days), size=n, p=day_w / day_w.sum())
    sys_idx = rng.choice(len(systems), size=n, p=sys_w / sys_w.sum())

    tickets = []
    for d_i, s_i in zip(day_idx, sys_idx, strict=True):
        opened = days[d_i].date()
        prefix = systems.at[s_i, "prefix"]
        probs = _category_probs(prefix, opened.month)
        tickets.append(
            {
                "ticket_key": len(tickets),
                "parent_key": None,
                "depth": 0,
                "sys_idx": int(s_i),
                "date_opened": opened,
                "fault_category": config.CATEGORIES[rng.choice(len(probs), p=probs)],
            }
        )

    # Repeat faults: the same system fails again in the same way within a few weeks.
    queue = list(tickets)
    while queue:
        parent = queue.pop()
        lemon = systems.at[parent["sys_idx"], "is_lemon"]
        p = scenario.lemon_recurrence_prob if lemon else scenario.recurrence_prob
        if rng.random() < p * 0.5 ** parent["depth"]:
            child_date = parent["date_opened"] + timedelta(days=int(rng.integers(3, 29)))
            if child_date <= config.END_DATE:
                child = {
                    **parent,
                    "ticket_key": len(tickets),
                    "parent_key": parent["ticket_key"],
                    "depth": parent["depth"] + 1,
                    "date_opened": child_date,
                }
                tickets.append(child)
                queue.append(child)

    slow = dict(scenario.slow_vendors)
    rows = []
    for t in tickets:
        sys = systems.iloc[t["sys_idx"]]
        opened: date = t["date_opened"]
        category = t["fault_category"]
        severity = config.SEVERITIES[rng.choice(4, p=SEVERITY_PROBS)]

        # Resolution time: severity baseline x vendor x site, with heavy lognormal noise.
        mult = slow.get(sys["vendor"], 1.0)
        if sys["site"] == scenario.backlog_site:
            mult *= scenario.backlog_slowdown
        days_to_close = SEVERITY_MEDIAN_DAYS[severity] * mult * rng.lognormal(0, RESOLUTION_SIGMA)
        long_tail = sys["vendor"] == scenario.long_tail_vendor and (
            rng.random() < scenario.long_tail_prob
        )
        if long_tail:
            days_to_close += int(rng.integers(60, 121))  # waiting for spare parts
        closed = opened + timedelta(days=int(round(days_to_close)))

        stuck_p = (
            scenario.backlog_stuck_prob
            if sys["site"] == scenario.backlog_site
            else scenario.stuck_prob
        )
        stuck = rng.random() < stuck_p
        if stuck or closed > config.AS_OF:
            age = (config.AS_OF - opened).days
            status = "In Progress" if age > 2 and rng.random() < 0.6 else "Open"
            closed = None
        else:
            status = "Closed"

        # Description noise: vague, two categories, or text contradicting the logged category.
        r = rng.random()
        others = [c for c in config.CATEGORIES if c != category]
        secondary = None
        logged = category
        if r < scenario.vague_rate:
            kind = "vague"
        elif r < scenario.vague_rate + scenario.multi_rate:
            kind = "multi"
            secondary = _pick(rng, others)
        elif r < scenario.vague_rate + scenario.multi_rate + scenario.mismatch_rate:
            kind = "mismatch"
            logged = _pick(rng, others)  # reporter picked the wrong category
        else:
            kind = "clean"
        text = describe(
            rng,
            category,
            "clean" if kind == "mismatch" else kind,
            secondary=secondary,
            summer=opened.month in SUMMER_MONTHS,
            recurrence=t["parent_key"] is not None,
        )

        rows.append(
            {
                "ticket_key": t["ticket_key"],
                "parent_key": t["parent_key"],
                # Tickets are entered into the system up to ~1.5 days after the fault.
                "entry_ts": opened.toordinal() + rng.random() * 1.5,
                "date_opened": opened,
                "date_closed": closed,
                "site": sys["site"],
                "system_type": sys["system_type"],
                "system_id": sys["system_id"],
                "vendor": sys["vendor"],
                "fault_category": category,
                "logged_category": logged,
                "description_type": kind,
                "secondary_category": secondary,
                "severity": severity,
                "description": text,
                "reported_by": _pick(rng, config.REPORTERS[sys["site"]]),
                "status": status,
                "is_lemon_system": bool(sys["is_lemon"]),
                "is_recurrence": t["parent_key"] is not None,
                "long_tail": long_tail,
                "stuck": stuck,
            }
        )
    return pd.DataFrame(rows)


def generate_dataset(scenario: Scenario | None = None) -> MessyDataset:
    scenario = scenario or Scenario()
    rng = np.random.default_rng(scenario.seed)
    truth = simulate_truth(scenario, rng)
    dataset = make_messy(truth, rng)
    dataset.summary["scenario"] = asdict(scenario)
    return dataset


def write_dataset(
    dataset: MessyDataset,
    raw_path: Path = config.RAW_REPORTS,
    truth_dir: Path = config.TRUTH_DIR,
) -> None:
    """Raw CSV goes to data/raw; everything that reveals the answers goes to data/truth."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    dataset.raw.to_csv(raw_path, index=False)
    dataset.truth.to_csv(truth_dir / "ground_truth.csv", index=False)
    dataset.duplicates.to_csv(truth_dir / "duplicates.csv", index=False)
    (truth_dir / "generation_summary.json").write_text(
        json.dumps(dataset.summary, indent=2, default=str)
    )


def main() -> None:
    dataset = generate_dataset()
    write_dataset(dataset)
    s = dataset.summary
    print(
        f"Generated {s['raw_rows']} raw rows ({s['unique_tickets']} real tickets) "
        f"-> {config.RAW_REPORTS.relative_to(config.ROOT)}"
    )
