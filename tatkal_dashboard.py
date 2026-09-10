"""
tatkal_dashboard.py

Streamlit UI for the Tatkal Booking Reform analysis tool.

Run with:
    streamlit run tatkal_dashboard.py

All calculation logic lives in tatkal_model.py -- this file is purely
presentation (inputs, tables, charts, report export).
"""

from datetime import datetime

import pandas as pd
import streamlit as st

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
            "Current peak-infrastructure cost (INR/day)",
            min_value=0.0, value=float(oh.current_infra_cost_per_day), step=100000.0,
            help="Cost of provisioning servers/gateways to survive the current burst window.",
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

    display_df = summary_df.set_index("Scenario").copy()
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

    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Revenue vs. overhead vs. net revenue (INR/day)**")
        chart_df = summary_df.set_index("Scenario")[
            ["Revenue (INR/day)", "Overhead (INR/day)", "Net revenue (INR/day)"]
        ]
        st.bar_chart(chart_df)

    with chart_col2:
        st.markdown("**Successful vs. unsuccessful applications/day**")
        vol_df = summary_df.set_index("Scenario")[["Successful/day", "Unsuccessful/day"]]
        st.bar_chart(vol_df)

    st.divider()
    st.markdown("**Denial rate (% of applicants who do not get a seat)**")
    st.bar_chart(summary_df.set_index("Scenario")[["Denial rate %"]])

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
        "Unsuccessful/day": round(tr.unsuccessful),
        "Applications/day": round(tr.applications),
        "Revenue (INR/day)": format_inr(tr.revenue),
    } for tr in picked.tier_results])
    st.dataframe(tier_detail_df, width='stretch')

# --------------------------------------------------------------------------
# Tab 3: Export report
# --------------------------------------------------------------------------

with tab_export:
    st.subheader("Export a summary report")
    st.markdown(
        "Generates a Markdown report listing every input variable and the "
        "three-scenario comparison, timestamped at generation time."
    )

    def build_report() -> str:
        lines = []
        lines.append("# Tatkal booking reform -- analysis report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(
            "\n> Figures below reflect user-entered inputs in this session. "
            "Defaults are illustrative placeholders, not verified IRCTC data."
        )

        lines.append("\n## 1. Input variables\n")
        lines.append("### 1.1 Demand, fare & charge structure by tier\n")
        lines.append(st.session_state.tiers_df.to_markdown(index=False))

        lines.append("\n### 1.2 System overhead assumptions\n")
        oh = st.session_state.overhead
        lines.append(f"- Current peak-infrastructure cost: {format_inr(oh.current_infra_cost_per_day)}/day")
        lines.append(f"- Avg. lock/payment attempts per confirmed seat: {oh.lock_release_multiplier:g}")
        lines.append(f"- Cost per failed payment attempt: INR {oh.payment_failure_cost:g}")
        lines.append(f"- Proposed peak-infrastructure cost: {format_inr(oh.proposed_infra_cost_per_day)}/day")
        lines.append(f"- Verification cost per application: INR {oh.verification_cost_per_application:g}")
        lines.append(f"- Refund/release cost per unsuccessful applicant: INR {oh.refund_processing_cost_per_unsuccessful:g}")
        lines.append(f"- Scenario C demand multiplier: {st.session_state.surge_multiplier:g}x")

        lines.append("\n## 2. Scenario comparison\n")
        report_summary_df = summary_df.copy()
        report_summary_df["Revenue (INR/day)"] = report_summary_df["Revenue (INR/day)"].apply(format_inr)
        report_summary_df["Overhead (INR/day)"] = report_summary_df["Overhead (INR/day)"].apply(format_inr)
        report_summary_df["Net revenue (INR/day)"] = report_summary_df["Net revenue (INR/day)"].apply(format_lakhs)
        lines.append(report_summary_df.to_markdown(index=False))

        lines.append("\n## 3. Key takeaways\n")
        rev_delta_b = proposed.net_revenue - current.net_revenue
        rev_delta_c = proposed_surge.net_revenue - current.net_revenue
        lines.append(
            f"- Moving from Current to Proposed (same demand) changes net revenue "
            f"by {format_lakhs(rev_delta_b)}/day."
        )
        lines.append(
            f"- If demand rises {st.session_state.surge_multiplier:g}x post-reform (Scenario C), "
            f"net revenue changes by {format_lakhs(rev_delta_c)}/day versus Current, "
            f"and overhead moves from {format_inr(current.total_overhead)}/day to "
            f"{format_inr(proposed_surge.total_overhead)}/day."
        )
        lines.append(
            f"- Denial rate moves from {current.denial_rate_pct:.1f}% (Current) to "
            f"{proposed.denial_rate_pct:.1f}% (Proposed, same demand) to "
            f"{proposed_surge.denial_rate_pct:.1f}% (Proposed, surge demand)."
        )

        lines.append("\n## 4. Tier-level detail (all scenarios)\n")
        for label, res in [("Current", current), ("Proposed (same demand)", proposed),
                            ("Proposed (demand surge)", proposed_surge)]:
            lines.append(f"\n### {label}\n")
            tdf = pd.DataFrame([{
                "Tier": tr.name, "Successful/day": round(tr.successful),
                "Unsuccessful/day": round(tr.unsuccessful),
                "Applications/day": round(tr.applications),
                "Revenue (INR/day)": format_inr(tr.revenue),
            } for tr in res.tier_results])
            lines.append(tdf.to_markdown(index=False))

        lines.append(
            "\n---\n*Generated by the Tatkal Reform Analysis Tool. "
            "See README.md for model assumptions, formulas, and extension points.*"
        )
        return "\n".join(lines)

    report_text = build_report()
    st.download_button(
        label="Download report (Markdown)",
        data=report_text,
        file_name=f"tatkal_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )

    with st.expander("Preview report"):
        st.markdown(report_text)
