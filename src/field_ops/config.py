"""Shared constants: paths, the reporting window and canonical reference lists.

Canonical lists live here (not in the cleaning code) because the generator, the cleaner
and the Excel report must all agree on what a "correct" value looks like.
"""

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRUTH_DIR = DATA_DIR / "truth"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUT_DIR = ROOT / "output"

RAW_REPORTS = RAW_DIR / "fault_reports.csv"


def clean_csv_path() -> Path:
    """The clean CSV the metrics, Excel report and dashboard all read (override for tests)."""
    return Path(os.environ.get("FIELD_OPS_CLEAN_CSV", OUTPUT_DIR / "clean_reports.csv"))


SEED = 42

# Excel stores dates as day counts from this epoch (1900 leap-year bug already accounted for).
EXCEL_EPOCH = date(1899, 12, 30)

# Repeat fault: a fault on the same system + category opening within this many days *after* the
# previous one was closed. Configurable (CLI: --repeat-window). The sensitivity table also shows
# SENSITIVITY_WINDOWS so the choice is visible.
REPEAT_WINDOW_DAYS = 14
SENSITIVITY_WINDOWS = (7, 14, 30)

# Reporting window. AS_OF is fixed (not "today") so aging metrics are reproducible.
START_DATE = date(2025, 10, 1)
END_DATE = date(2026, 9, 30)
AS_OF = END_DATE

VENDORS = [f"Vendor {letter}" for letter in "ABCDEFGHIJKLMNO"]

SITES = ["Northfield", "Harbor Point", "Lakeside", "Eastgate", "Westbrook", "Ridgeview"]

# Prefix used in system IDs -> full system type name.
SYSTEM_TYPES = {
    "CTL": "Control Unit",
    "COM": "Comms Module",
    "SEN": "Sensor Module",
    "PWR": "Power Unit",
    "FRM": "Mechanical Frame",
}

# Category given when the description gives no usable signal (never a guess).
UNCLASSIFIED = "Unclassified"

CATEGORIES = ["Power", "Communications", "Calibration", "Sensor", "Software", "Mechanical"]

SEVERITIES = ["Low", "Medium", "High", "Critical"]

STATUSES = ["Open", "In Progress", "Closed"]

# Generic civilian names, five field technicians per site.
REPORTERS = {
    "Northfield": ["Alex Morgan", "Dana Levi", "Sam Carter", "Maya Brooks", "Omar Haddad"],
    "Harbor Point": ["Noa Peretz", "Chris Walsh", "Lena Fischer", "Tom Reyes", "Yael Katz"],
    "Lakeside": ["Ben Avraham", "Priya Nair", "Jordan Lee", "Rita Gomes", "Eli Shapiro"],
    "Eastgate": ["Mia Novak", "David Cohen", "Sara Klein", "Ivan Petrov", "Tal Mizrahi"],
    "Westbrook": ["Kim Tanaka", "Ruth Adler", "Leo Martin", "Hana Saleh", "Nir Golan"],
    "Ridgeview": ["Amit Bar", "Grace Kim", "Paul Dubois", "Shira Ofer", "Max Weber"],
}
