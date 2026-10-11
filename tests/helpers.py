"""Independent pandas calculations used as the oracle for the Excel and dashboard tests."""

from field_ops import metrics


def expected_kpis(
    m: metrics.Metrics,
    vendors: list[str] | None = None,
    sites: list[str] | None = None,
    severities: list[str] | None = None,
) -> dict:
    """Headline numbers for a selection, computed straight from the prepared table.

    Deliberately does not call ``metrics.kpis`` / ``metrics.dashboard``: the tests compare those
    (and the Excel formulas) against this.
    """
    p, rows = m.prepared, m.repeat.rows
    mask = p["is_resolved"] | ~p["is_resolved"]  # all True, same index
    for column, allowed in (("vendor", vendors), ("site", sites), ("severity", severities)):
        if allowed is not None:
            mask &= p[column].isin(allowed)
    sub, resolved = p[mask], p[mask & p["is_resolved"]]
    checked = resolved["sla_target_days"].notna().sum()
    counted = rows.loc[mask, "counted"].sum()
    return {
        "tickets": len(sub),
        "resolved": len(resolved),
        "open": int(sub["is_open"].sum()),
        "open30": int((sub["age_days"] > 30).sum()),
        "median": resolved["days_to_resolve"].median() if len(resolved) else None,
        "mean": resolved["days_to_resolve"].mean() if len(resolved) else None,
        "breach": resolved["sla_breach"].sum() / checked if checked else None,
        "repeat": rows.loc[mask, "is_repeat"].sum() / counted if counted else None,
    }
