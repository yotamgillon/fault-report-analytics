"""Streamlit dashboard for the field operations metrics.

This file only renders. Every number, table and label of data comes from
``field_ops.metrics.dashboard()``, the same functions that build the Excel report, so the two
cannot disagree. (tests/test_dashboard.py checks that this file does no calculations.)

Run it with:  make dashboard
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from field_ops import config, metrics

BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#e1e0d9"  # validated categorical slots 1 and 2
SLA_NOTE = (
    "SLA targets and breach rates are **demo values** from `data/reference/vendors.csv` "
    "(synthetic, arbitrary tiers), not real contract terms."
)


@st.cache_resource(show_spinner="Computing metrics...")
def load_metrics(csv_path: str, modified: float) -> metrics.Metrics:
    """Read the clean CSV once per file version (``modified`` is only part of the cache key)."""
    return metrics.compute_all(
        metrics.load_clean_csv(csv_path),
        config.AS_OF,
        config.REPEAT_WINDOW_DAYS,
        metrics.load_vendors(),
    )


def show(value: float, spec: str) -> str:
    return "–" if pd.isna(value) else format(value, spec)


def line_chart(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, name, color in (
        ("opened", "Opened", BLUE),
        ("closed", "Closed (resolved)", ORANGE),
    ):
        fig.add_scatter(
            x=monthly["month"], y=monthly[column], name=name, mode="lines+markers",
            line={"color": color, "width": 2.5}, marker={"size": 8, "color": color},
        )  # fmt: skip
    fig.update_layout(legend={"orientation": "h", "y": -0.2}, yaxis_title="tickets")
    return fig


def median_chart(monthly: pd.DataFrame) -> go.Figure:
    symbols = ["circle-open" if low else "circle" for low in monthly["low_sample"]]
    fig = go.Figure(
        go.Scatter(
            x=monthly["month"], y=monthly["median_days"], mode="lines+markers",
            line={"color": BLUE, "width": 2.5},
            marker={"size": 9, "color": BLUE, "symbol": symbols},
            customdata=monthly["closed"],
            hovertemplate="%{x}<br>median %{y:.1f} days<br>n = %{customdata} closed<extra></extra>",
        )
    )  # fmt: skip
    fig.update_layout(yaxis_title="days", showlegend=False)
    fig.update_yaxes(rangemode="tozero")  # honest scale: a 5 -> 8 day change is not a cliff
    return fig


def aging_chart(aging: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=aging["bucket"], y=aging["tickets"], marker_color=BLUE,
                           text=aging["tickets"], textposition="outside"))  # fmt: skip
    fig.update_layout(yaxis_title="open tickets", showlegend=False)
    return fig


def repeat_chart(top: pd.DataFrame, fleet_rate: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=top["repeat_rate"], y=top["system_id"], orientation="h", marker_color=BLUE,
            texttemplate="%{x:.1%}", textposition="outside", cliponaxis=False,
            customdata=top[["counted_faults", "repeats", "vs_fleet_x"]],
            hovertemplate=("%{y}: %{x:.1%}<br>%{customdata[1]} of %{customdata[0]} faults"
                           "<br>%{customdata[2]:.2f}x fleet<extra></extra>"),
        )
    )  # fmt: skip
    fig.add_vline(x=fleet_rate, line_dash="dash", line_color=ORANGE)
    fig.update_layout(
        yaxis={"autorange": "reversed"},
        xaxis={
            "tickformat": ".0%",
            "title": f"repeat rate (dashed line: fleet {show(fleet_rate, '.1%')})",
        },
        showlegend=False,
    )
    return fig


def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis_gridcolor=GRID, yaxis_gridcolor=GRID)  # fmt: skip
    return fig


def grey_low_sample(table: pd.DataFrame):
    """Low-sample rows are greyed, like the Excel."""
    grey = "color: #898781; font-style: italic"
    return table.style.apply(lambda row: [grey if row["Sample"] else ""] * len(row), axis=1)


def flag_low_sample(card: pd.DataFrame) -> pd.DataFrame:
    return card.assign(low_sample=card["low_sample"].map({True: "low sample", False: ""}))


def scorecard_table(card: pd.DataFrame) -> tuple:
    names = {
        "vendor": "Vendor", "tickets": "Tickets", "resolved_n": "Resolved (n)",
        "median_days": "Median days", "mean_days": "Mean days", "low_sample": "Sample",
        "repeat_rate_pct": "Repeat rate", "vs_fleet_x": "Repeat vs fleet", "open": "Open",
        "over_30_days": "Open > 30 d",
    }  # fmt: skip
    table = flag_low_sample(card)[list(names)].rename(columns=names)
    config_ = {
        "Median days": st.column_config.NumberColumn(format="%.1f"),
        "Mean days": st.column_config.NumberColumn(format="%.1f"),
        "Repeat rate": st.column_config.NumberColumn(format="%.1f%%"),
        "Repeat vs fleet": st.column_config.NumberColumn(format="%.2fx"),
    }
    return grey_low_sample(table), config_


def sla_table(card: pd.DataFrame) -> tuple:
    names = {
        "vendor": "Vendor", "tier": "Tier (demo)", "sla_target_days": "SLA target, days (demo)",
        "resolved_n": "Resolved (n)", "sla_breaches": "SLA breaches (demo)",
        "sla_breach_rate_pct": "SLA breach rate (demo)", "low_sample": "Sample",
    }  # fmt: skip
    table = flag_low_sample(card)[list(names)].rename(columns=names)
    config_ = {"SLA breach rate (demo)": st.column_config.NumberColumn(format="%.1f%%")}
    return grey_low_sample(table), config_


def top_table(top: pd.DataFrame) -> tuple:
    names = {
        "rank": "Rank", "system_id": "System", "vendor": "Vendor", "site": "Site",
        "counted_faults": "Faults counted (n)", "repeats": "Repeats",
        "repeat_rate_pct": "Repeat rate", "vs_fleet_pts": "vs fleet (pts)",
        "vs_fleet_x": "vs fleet (x)", "tickets": "Tickets", "median_days": "Median days",
        "resolved_n": "Resolved (n)",
    }  # fmt: skip
    config_ = {
        "Repeat rate": st.column_config.NumberColumn(format="%.1f%%"),
        "vs fleet (pts)": st.column_config.NumberColumn(format="%+.1f"),
        "vs fleet (x)": st.column_config.NumberColumn(format="%.2fx"),
        "Median days": st.column_config.NumberColumn(format="%.1f"),
    }
    return top[list(names)].rename(columns=names), config_


def kpi_card(column, label: str, value: str, delta: str, detail: str, help_: str | None = None):
    """A metric with a short label/delta (never truncated) and the full explanation below it."""
    column.metric(label, value, delta, delta_color="off", border=True, help=help_)
    column.caption(detail)


st.set_page_config(page_title="Field Operations Analytics", layout="wide")

csv_path = config.clean_csv_path()
if not csv_path.exists():
    st.error(f"No clean data found at `{csv_path}`. Run `make all` first to build it.")
    st.stop()
data = load_metrics(str(csv_path), Path(csv_path).stat().st_mtime)
options = metrics.filter_options(data)

with st.sidebar:
    st.header("Filters")
    sites = st.multiselect("Site", options["sites"], default=options["sites"], key="site_filter")
    vendors = st.multiselect("Vendor", options["vendors"], default=options["vendors"],
                             key="vendor_filter")  # fmt: skip
    severities = st.multiselect("Severity", options["severities"], default=options["severities"],
                                key="severity_filter")  # fmt: skip
    st.caption("Repeat faults and the fleet average are always computed on all tickets.")

view = metrics.dashboard(
    data, metrics.Filters(vendors=tuple(vendors), sites=tuple(sites), severities=tuple(severities))
)
k, fleet = view.kpis, view.fleet

st.title("Field Operations Analytics")
st.caption(
    f"Reporting date {view.as_of} | {k['tickets']:,} of {view.n_total:,} tickets selected | "
    "synthetic data"
)
if view.empty:
    st.info("No tickets match the selected filters. Select at least one site, vendor and severity.")
    st.stop()

cards = st.columns(5)
kpi_card(cards[0], "Open faults", f"{k['open']:,}",
         f"{k['open_over_30_days']:,} older than 30 days",
         "Status Open or In Progress, aged against the reporting date.")  # fmt: skip
kpi_card(cards[1], "MTTR, median (days)", show(k["median_days"], ".1f"),
         f"n = {k['resolved']:,} resolved",
         f"Mean {show(k['mean_days'], '.1f')} days sits beside the median (right-skewed). "
         f"Fleet median {show(fleet['median_days'], '.1f')} days.")  # fmt: skip
kpi_card(cards[2], "Repeat rate", show(k["repeat_rate"], ".1%"),
         f"{k['repeats']:,} of {k['repeat_counted']:,} faults",
         f"{view.repeat_window}-day window after close. "
         f"Fleet {show(fleet['repeat_rate'], '.1%')}.")  # fmt: skip
kpi_card(cards[3], "SLA breach (demo)", show(k["sla_breach_rate"], ".1%"),
         f"{k['sla_breaches']:,} of {k['sla_checked']:,} resolved",
         "Against demo SLA targets (synthetic), not real contract terms. "
         f"Fleet {show(fleet['sla_breach_rate'], '.1%')}.")  # fmt: skip
kpi_card(cards[4], "Tickets", f"{k['tickets']:,}", f"of {view.n_total:,} in total",
         "Selected by the filters in the sidebar.")  # fmt: skip
if k["low_sample"]:
    st.warning(
        f"Low sample: fewer than {view.min_sample} resolved tickets in this selection, "
        "so treat the medians with caution."
    )
st.caption(SLA_NOTE)

st.subheader("Vendor scorecard")
performance_tab, sla_tab = st.tabs(["Performance", "SLA (demo values)"])
with performance_tab:
    styled, card_config = scorecard_table(view.scorecard)
    st.dataframe(styled, column_config=card_config, hide_index=True, width="stretch",
                 height="content")  # fmt: skip
with sla_tab:
    st.caption(SLA_NOTE)
    styled, sla_config = sla_table(view.scorecard)
    st.dataframe(styled, column_config=sla_config, hide_index=True, width="stretch",
                 height="content")  # fmt: skip
st.caption(
    "Median first, mean beside it; n = resolved tickets behind the numbers. "
    f"Groups under {view.min_sample} resolved tickets are marked low sample and shown greyed."
)

st.subheader("Trend")
left, right = st.columns(2)
with left:
    st.markdown("**Faults opened vs closed per month**")
    st.plotly_chart(style(line_chart(view.monthly)), width="stretch")
with right:
    st.markdown("**Median days to resolve, by month closed**")
    st.plotly_chart(style(median_chart(view.monthly)), width="stretch")
    st.caption(f"Hollow markers: fewer than {view.min_sample} tickets closed that month.")

st.subheader("Backlog and repeat faults")
left, right = st.columns(2)
with left:
    st.markdown("**Open backlog by age**")
    st.plotly_chart(style(aging_chart(view.aging)), width="stretch")
with right:
    st.markdown("**Top repeat-offender systems vs the fleet average**")
    if view.top_systems.empty:
        st.info(f"No system has at least {view.min_sample} counted faults in this selection.")
    else:
        st.plotly_chart(style(repeat_chart(view.top_systems, fleet["repeat_rate"])),
                        width="stretch")  # fmt: skip
if not view.top_systems.empty:
    top, top_config = top_table(view.top_systems)
    st.dataframe(top, column_config=top_config, hide_index=True, width="stretch")

with st.expander("Definitions"):
    for term, text in metrics.definitions(view.repeat_window):
        st.markdown(f"**{term}.** {text}")
