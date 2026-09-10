"""
tatkal_dashboard.py

Streamlit UI for the Tatkal Booking Reform analysis tool.

Run with:
    streamlit run tatkal_dashboard.py

All calculation logic lives in tatkal_model.py -- this file is purely
presentation (inputs, tables, charts, report export).
"""

import base64
import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# DejaVu Sans (bundled with matplotlib) has a glyph for the Indian Rupee
# sign (U+20B9); the standard PDF Helvetica font does not, so register it
# for use in report text and tables.
_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(_FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily(
    "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
    italic="DejaVuSans", boldItalic="DejaVuSans-Bold",
)
plt.rcParams["font.family"] = "DejaVu Sans"

from tatkal_model import (
    TierInput,
    OverheadAssumptions,
    run_all_scenarios,
    default_tiers,
    default_overhead,
)

st.set_page_config(page_title="Tatkal Reform Analysis Tool", layout="wide")

# --------------------------------------------------------------------------
# Indian-convention number formatting
# --------------------------------------------------------------------------
# Indian digit grouping: last 3 digits together, then groups of 2
# (e.g. 13,225,000 -> 1,32,25,000). Net revenue figures are additionally
# expressed in lakhs (1 lakh = 100,000).

def indian_grouping(value: float, decimals: int = 0) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    int_part = f"{value:.{decimals}f}"
    dec_part = ""
    if decimals > 0:
        int_part, dec_part = int_part.split(".")
        dec_part = "." + dec_part
    if len(int_part) > 3:
        last3, rest = int_part[-3:], int_part[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        int_part = ",".join(groups) + "," + last3
    return f"{sign}{int_part}{dec_part}"


def format_inr(value: float) -> str:
    return f"₹{indian_grouping(value)}"


def format_lakhs(value: float) -> str:
    return f"₹{indian_grouping(value / 100_000, decimals=2)} L"


# --------------------------------------------------------------------------
# Shared chart builder (used by both the dashboard tab and the PDF report)
# --------------------------------------------------------------------------

def build_stream_figure(summary_df: pd.DataFrame, figsize=(9, 3.6), title: str = None):
    """Revenue-by-stream chart with per-segment value/% labels and a full,
    untruncated legend -- st.bar_chart can't do either (Vega-Lite truncates
    long legend text with no way to widen it, and has no data-label option).
    """
    stream_df = summary_df.set_index("Scenario")[
        ["Flat fee revenue (INR/day)", "Charge/premium revenue (INR/day)"]
    ]
    flat = stream_df["Flat fee revenue (INR/day)"] / 100_000
    charge = stream_df["Charge/premium revenue (INR/day)"] / 100_000
    totals = flat + charge

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(stream_df))
    ax.barh(y_pos, flat, color="#7c3aed", label="Flat fee revenue (₹ lakh/day)")
    ax.barh(y_pos, charge, left=flat, color="#0891b2", label="Charge/premium revenue (₹ lakh/day)")

    for i, (f, c, total) in enumerate(zip(flat, charge, totals)):
        if f > 0:
            ax.text(f / 2, i, f"₹{f:,.0f} L\n({f / total * 100:.0f}%)",
                     va="center", ha="center", color="white", fontsize=7.5)
        if c > 0:
            ax.text(f + c / 2, i, f"₹{c:,.0f} L\n({c / total * 100:.0f}%)",
                     va="center", ha="center", color="white", fontsize=7.5)
        ax.text(total + totals.max() * 0.015, i, f"₹{total:,.0f} L total",
                 va="center", ha="left", fontsize=7.5, color="#374151")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(stream_df.index)
    ax.invert_yaxis()  # first scenario (Current) at top, matching the table order
    ax.set_xlim(0, totals.max() * 1.18)  # headroom for the "total" labels
    ax.set_xlabel("₹ lakh/day")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def build_revenue_figure(summary_df: pd.DataFrame, figsize=(9, 3.8), title: str = None):
    """Revenue vs. overhead vs. net revenue, as grouped (not stacked) bars
    with a value label on every bar and a full, untruncated legend below
    the chart -- st.bar_chart can't show labels and truncates long legend
    text with no way to widen it.
    """
    cols = ["Revenue (INR/day)", "Overhead (INR/day)", "Net revenue (INR/day)"]
    labels = ["Revenue (₹ lakh/day)", "Overhead (₹ lakh/day)", "Net revenue (₹ lakh/day)"]
    plot_colors = ["#2563eb", "#dc2626", "#16a34a"]
    chart_df = summary_df.set_index("Scenario")[cols] / 100_000

    n_scenarios = len(chart_df)
    n_series = len(cols)
    bar_h = 0.8 / n_series
    y_base = list(range(n_scenarios))
    x_max = chart_df.to_numpy().max()

    fig, ax = plt.subplots(figsize=figsize)
    for s, (col, label, color) in enumerate(zip(cols, labels, plot_colors)):
        offset = (s - (n_series - 1) / 2) * bar_h
        y = [yb + offset for yb in y_base]
        values = chart_df[col].tolist()
        ax.barh(y, values, height=bar_h, color=color, label=label)
        for yy, v in zip(y, values):
            ax.text(v + x_max * 0.015, yy, f"₹{v:,.0f} L", va="center", ha="left", fontsize=7, color="#374151")

    ax.set_yticks(y_base)
    ax.set_yticklabels(chart_df.index)
    ax.invert_yaxis()  # first scenario (Current) at top, matching the table order
    ax.set_xlim(0, x_max * 1.22)  # headroom for the value labels
    ax.set_xlabel("₹ lakh/day")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def build_volume_figure(summary_df: pd.DataFrame, figsize=(9, 3.6), title: str = None):
    """Successful / unsuccessful / denied-access applications, stacked
    (they genuinely sum to total demand) with per-segment value/% labels,
    a total-per-bar label, and a full, untruncated legend below the chart.
    """
    cols = ["Successful/day", "Unsuccessful (lost seat)/day", "Denied access (locked out)/day"]
    labels = ["Successful (lakh/day)", "Unsuccessful, lost seat (lakh/day)",
              "Denied access, locked out (lakh/day)"]
    plot_colors = ["#16a34a", "#f59e0b", "#dc2626"]
    vol_df = summary_df.set_index("Scenario")[cols] / 100_000
    totals = vol_df.sum(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(vol_df))
    left = [0.0] * len(vol_df)
    for col, label, color in zip(cols, labels, plot_colors):
        values = vol_df[col].tolist()
        ax.barh(y_pos, values, left=left, color=color, label=label)
        for i, (v, l, total) in enumerate(zip(values, left, totals)):
            if v > 0:
                ax.text(l + v / 2, i, f"{v:,.1f} L\n({v / total * 100:.0f}%)",
                         va="center", ha="center", color="white", fontsize=7.5)
        left = [l + v for l, v in zip(left, values)]

    for i, total in enumerate(totals):
        ax.text(total + totals.max() * 0.015, i, f"{total:,.1f} L total",
                 va="center", ha="left", fontsize=7.5, color="#374151")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(vol_df.index)
    ax.invert_yaxis()  # first scenario (Current) at top, matching the table order
    ax.set_xlim(0, totals.max() * 1.18)  # headroom for the "total" labels
    ax.set_xlabel("lakh/day")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Session-state init
# --------------------------------------------------------------------------

if "tiers_df" not in st.session_state:
    tiers = default_tiers()
    st.session_state.tiers_df = pd.DataFrame([{
        "Tier": t.name,
        "Seats/day": t.seats_per_day,
        "Requests/day": t.requests_per_day,
        "Avg base fare (INR)": t.avg_base_fare,
        "Current surcharge %": t.current_surcharge_pct,
        "Current min (INR)": t.current_min_charge,
        "Current max (INR)": t.current_max_charge,
        "Proposed flat fee (INR)": t.proposed_flat_fee,
        "Proposed premium %": t.proposed_premium_pct,
        "Proposed premium min (INR)": t.proposed_min_premium,
        "Proposed premium max (INR)": t.proposed_max_premium,
    } for t in tiers])

if "overhead" not in st.session_state:
    st.session_state.overhead = default_overhead()

if "surge_multiplier" not in st.session_state:
    st.session_state.surge_multiplier = 3.0


def df_to_tiers(df: pd.DataFrame):
    return [
        TierInput(
            name=row["Tier"],
            seats_per_day=float(row["Seats/day"]),
            requests_per_day=float(row["Requests/day"]),
            avg_base_fare=float(row["Avg base fare (INR)"]),
            current_surcharge_pct=float(row["Current surcharge %"]),
            current_min_charge=float(row["Current min (INR)"]),
            current_max_charge=float(row["Current max (INR)"]),
            proposed_flat_fee=float(row["Proposed flat fee (INR)"]),
            proposed_premium_pct=float(row["Proposed premium %"]),
            proposed_min_premium=float(row["Proposed premium min (INR)"]),
            proposed_max_premium=float(row["Proposed premium max (INR)"]),
        )
        for _, row in df.iterrows()
    ]


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Tatkal booking reform: analysis dashboard")
st.caption(
    "Compare today's FCFS Tatkal system against a proposed computerized "
    "random allotment system, including a scenario where application "
    "volume rises after reform while seat availability stays fixed. "
    "All figures are user-editable illustrative inputs -- see README.md."
)

tab_inputs, tab_compare, tab_export = st.tabs(
    ["1. Inputs", "2. Scenario comparison", "3. Export report"]
)

# --------------------------------------------------------------------------
# Tab 1: Inputs
# --------------------------------------------------------------------------

with tab_inputs:
    st.subheader("Demand, fare & charge structure by tier")
    st.markdown(
        "Edit any cell directly. `Seats/day` is fixed Tatkal capacity; "
        "`Requests/day` is today's observed demand. Successful / "
        "unsuccessful bookings are derived automatically as "
        "`min(seats, requests)` and the remainder."
    )
    st.session_state.tiers_df = st.data_editor(
        st.session_state.tiers_df,
        num_rows="dynamic",
        width='stretch',
        key="tiers_editor",
    )

    st.divider()
    st.subheader("System overhead assumptions")
    st.markdown(
        "These drive the cost side of the comparison. Defaults are "
        "illustrative placeholders -- replace with real cost estimates "
        "when available."
    )

    oh = st.session_state.overhead
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current system**")
        oh.current_infra_cost_per_day = st.number_input(
            "Current actual infrastructure spend (INR/day)",
            min_value=0.0, value=float(oh.current_infra_cost_per_day), step=100000.0,
            help="What's actually spent on servers/gateways for the current burst window.",
        )
        oh.current_infra_required_cost_per_day = st.number_input(
            "Infra spend required for 100% access (INR/day)",
            min_value=0.0, value=float(oh.current_infra_required_cost_per_day), step=100000.0,
            help="What it would cost to give every applicant access to the booking "
                 "process during the burst -- i.e. nobody turned away by an "
                 "overloaded server. The gap between this and actual spend above "
                 "determines how many applicants are denied access outright.",
        )
        _capacity_pct = (
            min(1.0, oh.current_infra_cost_per_day / oh.current_infra_required_cost_per_day)
            if oh.current_infra_required_cost_per_day > 0 else 1.0
        )
        st.caption(
            f"→ Current system infra capacity: **{_capacity_pct * 100:.0f}%** of demand can "
            f"even enter the process at today's spend. The rest are denied access outright, "
            f"separate from those who enter and lose the seat lottery."
        )
        oh.lock_release_multiplier = st.number_input(
            "Avg. lock/payment attempts per confirmed seat",
            min_value=1.0, value=float(oh.lock_release_multiplier), step=0.1,
            help="How many times a seat is locked & payment attempted (incl. failures) before one booking succeeds. 1.0 = no repeated locking.",
        )
        oh.payment_failure_cost = st.number_input(
            "Cost per failed payment attempt (INR)",
            min_value=0.0, value=float(oh.payment_failure_cost), step=1.0,
            help="Support / reconciliation cost for each 'amount deducted, ticket not booked' style failure.",
        )

    with col2:
        st.markdown("**Proposed system**")
        oh.proposed_infra_cost_per_day = st.number_input(
            "Proposed peak-infrastructure cost (INR/day)",
            min_value=0.0, value=float(oh.proposed_infra_cost_per_day), step=100000.0,
            help="Cost of provisioning for a smoothed, extended-window arrival pattern instead of a burst.",
        )
        st.caption(
            "→ Proposed system infra capacity: **100%** by design -- spreading "
            "intake over an extended window instead of a single burst means "
            "nobody is denied access outright, at any demand level."
        )
        oh.verification_cost_per_application = st.number_input(
            "Verification / fraud-screening cost per application (INR)",
            min_value=0.0, value=float(oh.verification_cost_per_application), step=0.5,
            help="Charged once per application, win or lose.",
        )
        oh.refund_processing_cost_per_unsuccessful = st.number_input(
            "Refund / block-release cost per unsuccessful applicant (INR)",
            min_value=0.0, value=float(oh.refund_processing_cost_per_unsuccessful), step=0.5,
        )

    st.divider()
    st.subheader("Demand-surge scenario")
    st.session_state.surge_multiplier = st.slider(
        "Scenario 3 demand multiplier (applications relative to today, seats unchanged)",
        min_value=1.0, max_value=10.0, value=float(st.session_state.surge_multiplier), step=0.5,
        help="Models more people applying because entry is cheap/low-risk under the proposed system, while capacity stays the same.",
    )

# --------------------------------------------------------------------------
# Compute scenarios (used by both remaining tabs)
# --------------------------------------------------------------------------

tiers = df_to_tiers(st.session_state.tiers_df)
overhead = st.session_state.overhead
results = run_all_scenarios(tiers, overhead, surge_multiplier=st.session_state.surge_multiplier)
current, proposed, proposed_surge = results["current"], results["proposed"], results["proposed_surge"]

summary_df = pd.DataFrame([current.as_dict(), proposed.as_dict(), proposed_surge.as_dict()])

# --------------------------------------------------------------------------
# Tab 2: Scenario comparison
# --------------------------------------------------------------------------

with tab_compare:
    st.subheader("Three-scenario comparison")
    st.markdown(
        "**Scenario A** = current FCFS system today. "
        "**Scenario B** = proposed computerized random allotment at *today's* demand. "
        f"**Scenario C** = proposed system if applications rise to "
        f"**{st.session_state.surge_multiplier:g}x** today's level while seats stay fixed."
    )

    display_df = summary_df.drop(columns=["Denial rate %"]).set_index("Scenario").copy()
    display_df["Flat fee revenue (INR/day)"] = display_df["Flat fee revenue (INR/day)"].apply(format_inr)
    display_df["Charge/premium revenue (INR/day)"] = display_df["Charge/premium revenue (INR/day)"].apply(format_inr)
    display_df["Revenue (INR/day)"] = display_df["Revenue (INR/day)"].apply(format_inr)
    display_df["Overhead (INR/day)"] = display_df["Overhead (INR/day)"].apply(format_inr)
    display_df["Net revenue (INR/day)"] = display_df["Net revenue (INR/day)"].apply(format_lakhs)
    st.dataframe(display_df, width='stretch')

    c1, c2, c3 = st.columns(3)
    c1.metric("Net revenue -- Current", f"{format_lakhs(current.net_revenue)}/day")
    c2.metric("Net revenue -- Proposed", f"{format_lakhs(proposed.net_revenue)}/day",
              delta=f"{format_lakhs(proposed.net_revenue - current.net_revenue)}")
    c3.metric("Net revenue -- Proposed + surge", f"{format_lakhs(proposed_surge.net_revenue)}/day",
              delta=f"{format_lakhs(proposed_surge.net_revenue - current.net_revenue)}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Locked out of system -- Current", f"{current.access_denied_pct:.0f}%",
              help="Applicants denied access outright by infra capacity, before the seat lottery even runs.")
    c5.metric("Locked out of system -- Proposed", f"{proposed.access_denied_pct:.0f}%")
    c6.metric("Locked out of system -- Proposed + surge", f"{proposed_surge.access_denied_pct:.0f}%")

    st.divider()
    st.markdown("**Revenue vs. overhead vs. net revenue (₹ lakh/day)**")
    st.caption(
        "Bars are grouped, not stacked -- Net revenue = Revenue - Overhead, "
        "so stacking all three would double-count."
    )
    revenue_fig = build_revenue_figure(summary_df)
    st.pyplot(revenue_fig)
    plt.close(revenue_fig)

    st.divider()
    st.markdown("**Successful / unsuccessful / denied-access applications (lakh/day)**")
    st.caption(
        "'Unsuccessful' entered the process but lost the seat lottery. "
        "'Denied access' were turned away by the system itself -- infra "
        "capacity, not seat scarcity. The current system shows both; the "
        "proposed system is designed for zero denied access."
    )
    volume_fig = build_volume_figure(summary_df)
    st.pyplot(volume_fig)
    plt.close(volume_fig)

    st.divider()
    st.markdown("**Revenue by stream (₹ lakh/day)**")
    st.caption(
        "Flat fee is paid by every applicant, win or lose -- 0 under Current, "
        "which has no flat-fee stream. Charge/premium is paid only by successful "
        "bookers: the Tatkal surcharge under Current, the reduced premium under "
        "Proposed. The two bars are stacked because they genuinely sum to Revenue."
    )
    stream_fig = build_stream_figure(summary_df)
    st.pyplot(stream_fig)
    plt.close(stream_fig)

    st.divider()
    st.subheader("Tier-level detail")
    scenario_pick = st.selectbox(
        "View tier-level breakdown for:",
        options=["Current", "Proposed (same demand)", "Proposed (demand surge)"],
    )
    picked = {"Current": current, "Proposed (same demand)": proposed,
              "Proposed (demand surge)": proposed_surge}[scenario_pick]
    tier_detail_df = pd.DataFrame([{
        "Tier": tr.name,
        "Successful/day": round(tr.successful),
        "Unsuccessful (lost seat)/day": round(tr.unsuccessful),
        "Denied access (locked out)/day": round(tr.denied_access),
        "Applications/day": round(tr.applications),
        "Flat fee revenue (INR/day)": format_inr(tr.flat_fee_revenue),
        "Charge/premium revenue (INR/day)": format_inr(tr.charge_revenue),
        "Revenue (INR/day)": format_inr(tr.revenue),
    } for tr in picked.tier_results])
    st.dataframe(tier_detail_df, width='stretch')

# --------------------------------------------------------------------------
# Tab 3: Export report (PDF)
# --------------------------------------------------------------------------

with tab_export:
    st.subheader("Export a summary report")
    st.markdown(
        "Generates a PDF report listing every input variable, the "
        "three-scenario comparison table, the comparison charts from the "
        "**Scenario comparison** tab, and tier-level detail -- timestamped "
        "at generation time."
    )

    def _fmt_num(v) -> str:
        v = float(v)
        return str(int(v)) if v == int(v) else f"{v:.2f}"

    def _df_to_table(df: pd.DataFrame, font_size: float = 7, col_widths=None) -> Table:
        # Wrap cell text in Paragraphs so long headers/values wrap inside a
        # fitted column width instead of overflowing past the page edge.
        header_style = ParagraphStyle(
            "th", fontName="DejaVuSans-Bold", fontSize=font_size,
            leading=font_size + 2, textColor=colors.white, alignment=TA_CENTER,
        )
        left_style = ParagraphStyle(
            "td_left", fontName="DejaVuSans", fontSize=font_size, leading=font_size + 2,
        )
        right_style = ParagraphStyle(
            "td_right", fontName="DejaVuSans", fontSize=font_size,
            leading=font_size + 2, alignment=TA_RIGHT,
        )
        header_row = [Paragraph(str(c), header_style) for c in df.columns]
        body_rows = []
        for row in df.astype(str).values.tolist():
            body_rows.append([
                Paragraph(v, left_style if i == 0 else right_style)
                for i, v in enumerate(row)
            ])
        data = [header_row] + body_rows
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        return table

    def _make_chart_image(fig, width_cm: float) -> Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        w_in, h_in = fig.get_size_inches()
        height_cm = width_cm * (h_in / w_in)
        return Image(buf, width=width_cm * cm, height=height_cm * cm)

    def _revenue_chart(summary_df: pd.DataFrame) -> Image:
        fig = build_revenue_figure(summary_df, figsize=(9, 3.8),
                                    title="Revenue vs. overhead vs. net revenue (₹ lakh/day)")
        return _make_chart_image(fig, width_cm=24)

    def _volume_chart(summary_df: pd.DataFrame) -> Image:
        fig = build_volume_figure(summary_df, figsize=(9, 3.6),
                                   title="Successful / unsuccessful / denied-access applications (lakh/day)")
        return _make_chart_image(fig, width_cm=24)

    def _stream_chart(summary_df: pd.DataFrame) -> Image:
        fig = build_stream_figure(summary_df, figsize=(9, 3.2), title="Revenue by stream (₹ lakh/day)")
        return _make_chart_image(fig, width_cm=24)

    def build_pdf_report() -> bytes:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=landscape(A4),
            leftMargin=1.2 * cm, rightMargin=1.2 * cm,
            topMargin=1.2 * cm, bottomMargin=1.2 * cm,
        )
        styles = getSampleStyleSheet()
        for name, bold in [("Title", True), ("Normal", False), ("Italic", False),
                            ("Heading1", True), ("Heading2", True)]:
            styles[name].fontName = "DejaVuSans-Bold" if bold else "DejaVuSans"
        story = []

        story.append(Paragraph("Tatkal booking reform -- analysis report", styles["Title"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            "Figures below reflect user-entered inputs in this session. "
            "Defaults are illustrative placeholders, not verified IRCTC data.",
            styles["Italic"],
        ))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("1. Input variables", styles["Heading1"]))
        story.append(Paragraph("1.1 Demand, fare & charge structure by tier", styles["Heading2"]))
        tiers_display = st.session_state.tiers_df.copy()
        for col in tiers_display.columns:
            if col != "Tier":
                tiers_display[col] = tiers_display[col].apply(_fmt_num)
        n_other_cols = len(tiers_display.columns) - 1
        tier_col_widths = [3.0 * cm] + [2.3 * cm] * n_other_cols
        story.append(_df_to_table(tiers_display, font_size=6.5, col_widths=tier_col_widths))
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph("1.2 System overhead assumptions", styles["Heading2"]))
        oh = st.session_state.overhead
        _capacity_pct_report = (
            min(1.0, oh.current_infra_cost_per_day / oh.current_infra_required_cost_per_day)
            if oh.current_infra_required_cost_per_day > 0 else 1.0
        )
        for line in [
            f"Current actual infrastructure spend: {format_inr(oh.current_infra_cost_per_day)}/day",
            f"Infra spend required for 100% access: {format_inr(oh.current_infra_required_cost_per_day)}/day "
            f"(→ current system infra capacity: {_capacity_pct_report * 100:.0f}%)",
            f"Avg. lock/payment attempts per confirmed seat: {oh.lock_release_multiplier:g}",
            f"Cost per failed payment attempt: INR {oh.payment_failure_cost:g}",
            f"Proposed peak-infrastructure cost: {format_inr(oh.proposed_infra_cost_per_day)}/day "
            f"(→ proposed system infra capacity: 100% by design)",
            f"Verification cost per application: INR {oh.verification_cost_per_application:g}",
            f"Refund/release cost per unsuccessful applicant: INR {oh.refund_processing_cost_per_unsuccessful:g}",
            f"Scenario C demand multiplier: {st.session_state.surge_multiplier:g}x",
        ]:
            story.append(Paragraph(f"&bull; {line}", styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("2. Scenario comparison", styles["Heading1"]))
        story.append(Paragraph(
            "Scenario A = current FCFS system today. Scenario B = proposed computerized "
            "random allotment at today's demand. Scenario C = proposed system if applications "
            f"rise to {st.session_state.surge_multiplier:g}x today's level while seats stay fixed.",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.3 * cm))
        report_summary_df = summary_df.drop(columns=["Denial rate %"]).copy()
        report_summary_df["Flat fee revenue (INR/day)"] = report_summary_df["Flat fee revenue (INR/day)"].apply(format_inr)
        report_summary_df["Charge/premium revenue (INR/day)"] = report_summary_df["Charge/premium revenue (INR/day)"].apply(format_inr)
        report_summary_df["Revenue (INR/day)"] = report_summary_df["Revenue (INR/day)"].apply(format_inr)
        report_summary_df["Overhead (INR/day)"] = report_summary_df["Overhead (INR/day)"].apply(format_inr)
        report_summary_df["Net revenue (INR/day)"] = report_summary_df["Net revenue (INR/day)"].apply(format_lakhs)
        story.append(_df_to_table(report_summary_df, font_size=6.5))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("Scenario comparison charts", styles["Heading2"]))
        story.append(_revenue_chart(summary_df))
        story.append(Spacer(1, 0.3 * cm))
        story.append(_volume_chart(summary_df))
        story.append(Spacer(1, 0.3 * cm))
        story.append(_stream_chart(summary_df))
        story.append(Paragraph(
            "Note on denial rate: under the current FCFS system this isn't a clean "
            "'sold out' notice -- heavy load at the burst window means most of these "
            "applicants are kept waiting or told to try again as seat locks are "
            "repeatedly grabbed and released, which is where the frustration comes from.",
            styles["Italic"],
        ))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("3. Key takeaways", styles["Heading1"]))
        rev_delta_b = proposed.net_revenue - current.net_revenue
        rev_delta_c = proposed_surge.net_revenue - current.net_revenue
        for line in [
            f"Moving from Current to Proposed (same demand) changes net revenue "
            f"by {format_lakhs(rev_delta_b)}/day.",
            f"If demand rises {st.session_state.surge_multiplier:g}x post-reform (Scenario C), "
            f"net revenue changes by {format_lakhs(rev_delta_c)}/day versus Current, "
            f"and overhead moves from {format_inr(current.total_overhead)}/day to "
            f"{format_inr(proposed_surge.total_overhead)}/day.",
            f"Under Current, {current.access_denied_pct:.1f}% of demand is denied "
            f"access outright by infra capacity (never got to compete for a seat) "
            f"and {current.seat_lottery_denial_pct:.1f}% lost the seat lottery after "
            f"getting in. The Proposed system is designed for 0% denied access at "
            f"any demand level -- everyone gets into the process.",
        ]:
            story.append(Paragraph(f"&bull; {line}", styles["Normal"]))

        story.append(PageBreak())
        story.append(Paragraph("4. Tier-level detail (all scenarios)", styles["Heading1"]))
        for label, res in [("Current", current), ("Proposed (same demand)", proposed),
                            ("Proposed (demand surge)", proposed_surge)]:
            story.append(Paragraph(label, styles["Heading2"]))
            tdf = pd.DataFrame([{
                "Tier": tr.name, "Successful/day": round(tr.successful),
                "Unsuccessful (lost seat)/day": round(tr.unsuccessful),
                "Denied access (locked out)/day": round(tr.denied_access),
                "Applications/day": round(tr.applications),
                "Flat fee revenue (INR/day)": format_inr(tr.flat_fee_revenue),
                "Charge/premium revenue (INR/day)": format_inr(tr.charge_revenue),
                "Revenue (INR/day)": format_inr(tr.revenue),
            } for tr in res.tier_results])
            story.append(_df_to_table(tdf, font_size=7))
            story.append(Spacer(1, 0.4 * cm))

        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            "Generated by the Tatkal Reform Analysis Tool.",
            styles["Italic"],
        ))

        doc.build(story)
        return buf.getvalue()

    pdf_bytes = build_pdf_report()
    st.download_button(
        label="Download report (PDF)",
        data=pdf_bytes,
        file_name=f"tatkal_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )

    with st.expander("Preview report"):
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="800" style="border:1px solid #ddd;"></iframe>',
            unsafe_allow_html=True,
        )
