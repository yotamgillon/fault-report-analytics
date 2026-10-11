"""Alias tables: labels that mean something but cannot be matched by spelling alone.

How these were built: profile the distinct raw values per column, then map the ones a
similarity score cannot resolve (abbreviations, synonyms such as "Network" -> Communications).
Plain misspellings are deliberately NOT listed for categories: the fuzzy matcher handles those,
so a new typo still resolves without anyone editing this file. Values that match neither end up
as "Unknown" and show up in the data-quality report, which is the signal to extend this table.
"""

CATEGORY_ALIASES = {
    "pwr": "Power",
    "power supply": "Power",
    "battery/power": "Power",
    "comm": "Communications",
    "comms": "Communications",
    "network": "Communications",
    "cal": "Calibration",
    "cal/align": "Calibration",
    "detector": "Sensor",
    "camera/sensor": "Sensor",
    "sw": "Software",
    "s/w": "Software",
    "software bug": "Software",
    "mech": "Mechanical",
    "frame/mech": "Mechanical",
}

# Severity and status are short closed lists, so they use alias tables only (no fuzzy matching):
# with 3-4 values a similarity score could turn "Hi" into the wrong level.
SEVERITY_ALIASES = {
    "l": "Low",
    "lo": "Low",
    "med": "Medium",
    "moderate": "Medium",
    "hi": "High",
    "hgh": "High",
    "crit": "Critical",
    "critcal": "Critical",
}

STATUS_ALIASES = {
    "new": "Open",
    "in-progress": "In Progress",
    "in prog": "In Progress",
    "wip": "In Progress",
    "resolved": "Closed",
    "done": "Closed",
}
