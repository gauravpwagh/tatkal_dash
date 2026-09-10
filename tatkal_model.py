"""
tatkal_model.py

Pure-Python calculation engine for the Tatkal Booking Reform analysis tool.
No UI dependencies here on purpose -- this module can be imported, unit
tested, or wired into a different frontend (Flask/FastAPI/React/etc.)
without touching any Streamlit code.

Core idea
---------
The system is modelled as a set of fare "tiers" (e.g. Non-AC, Mid-AC,
Upper-AC). For each tier we track:
  - how many Tatkal seats are available per day (fixed capacity)
  - how many booking requests come in per day (demand)
  - the average base fare for that tier

Two allocation/financing mechanisms are modelled:
  1. CURRENT  : today's system. Only successful bookers pay a
                percentage-of-fare Tatkal surcharge (with min/max caps).
                Overhead is driven by peak infrastructure cost plus the
                cost of repeated seat lock-and-release payment failures.
  2. PROPOSED : computerized random allotment. Every applicant pays a
                small flat, non-refundable processing fee; only
                successful applicants additionally pay a reduced,
                fare-proportional premium. Overhead is driven by a
                (lower) infrastructure cost plus a per-application
                verification cost and a per-unsuccessful-applicant
                refund/release processing cost.

A third scenario reuses the PROPOSED calculation with a demand
multiplier applied to requests, while seat availability (capacity)
stays fixed -- this models "more people apply because entry is cheap
and low-risk, but the number of seats doesn't change".

All monetary values are in INR. All "per day" figures are daily unless
otherwise stated.
"""

from dataclasses import dataclass, field
from typing import List, Dict


# --------------------------------------------------------------------------
# Input data structures
# --------------------------------------------------------------------------

@dataclass
class TierInput:
    """One fare tier's demand, fare, and charge-structure inputs."""
    name: str

    # Demand & fare
    seats_per_day: float          # Tatkal seats available per day (capacity)
    requests_per_day: float       # Tatkal booking requests/applications per day
    avg_base_fare: float          # average base fare (INR) for this tier

    # Current system: Tatkal surcharge = pct of base fare, clipped to [min, max]
    current_surcharge_pct: float
    current_min_charge: float
    current_max_charge: float

    # Proposed system: flat non-refundable processing fee (every applicant)
    proposed_flat_fee: float

    # Proposed system: reduced premium, winners only = pct of base fare, clipped
    proposed_premium_pct: float
    proposed_min_premium: float
    proposed_max_premium: float


@dataclass
class OverheadAssumptions:
    """System-wide cost assumptions, shared across all tiers."""

    # --- Current system overhead ---
    current_infra_cost_per_day: float
    # Average number of times a seat is locked (and possibly released back
    # to the pool) before a payment finally succeeds, under the current
    # burst/race mechanism. 1.0 = no repeated locking. See technical report
    # Section 2.1 ("seat lock-and-release cycle").
    lock_release_multiplier: float
    # Support / reconciliation cost incurred per failed payment attempt
    # ("amount deducted, ticket not booked" style failures).
    payment_failure_cost: float

    # --- Proposed system overhead ---
    proposed_infra_cost_per_day: float
    # Cost of running identity verification + fraud screening on ONE
    # application (charged whether or not that application wins the draw).
    verification_cost_per_application: float
    # Cost of releasing a block / processing a refund for ONE unsuccessful
    # applicant.
    refund_processing_cost_per_unsuccessful: float


# --------------------------------------------------------------------------
# Output data structures
# --------------------------------------------------------------------------

@dataclass
class TierResult:
    name: str
    successful: float
    unsuccessful: float
    applications: float
    revenue: float


@dataclass
class ScenarioResult:
    name: str
    description: str
    tier_results: List[TierResult]
    total_seats: float
    total_requests: float
    total_successful: float
    total_unsuccessful: float
    total_applications: float
    total_revenue: float
    total_overhead: float

    @property
    def net_revenue(self) -> float:
        return self.total_revenue - self.total_overhead

    @property
    def denial_rate_pct(self) -> float:
        """Share of applicants who do NOT get a seat.

        Under the current FCFS system this isn't a clean "sold out"
        message -- heavy load at the burst window means most of these
        applicants are kept waiting or told to try again as seat
        locks are repeatedly grabbed and released (see
        `lock_release_multiplier`), which is where the frustration
        comes from, not just the eventual failure to book.
        """
        if self.total_applications == 0:
            return 0.0
        return 100.0 * self.total_unsuccessful / self.total_applications

    def as_dict(self) -> Dict:
        return {
            "Scenario": self.name,
            "Total seats/day": round(self.total_seats),
            "Total requests/day": round(self.total_requests),
            "Successful/day": round(self.total_successful),
            "Unsuccessful/day": round(self.total_unsuccessful),
            "Denial rate %": round(self.denial_rate_pct, 1),
            "Revenue (INR/day)": round(self.total_revenue),
            "Overhead (INR/day)": round(self.total_overhead),
            "Net revenue (INR/day)": round(self.net_revenue),
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _clip(value: float, lo: float, hi: float) -> float:
    """Clip value to [lo, hi]. If lo > hi (bad input), just return value."""
    if lo > hi:
        return value
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------
# Core calculations
# --------------------------------------------------------------------------

def compute_current(tiers: List[TierInput], overhead: OverheadAssumptions) -> ScenarioResult:
    """Model today's FCFS system.

    Revenue is earned only from successful bookers, at the current
    percentage-of-fare surcharge (clipped to min/max).

    Overhead = flat peak-infrastructure cost + cost of failed payment
    attempts generated by the seat lock-and-release cycle (each confirmed
    seat is assumed to have gone through `lock_release_multiplier` payment
    attempts on average before succeeding; all attempts beyond the first
    successful one are treated as failures needing support/reconciliation).
    """
    tier_results = []
    total_seats = total_requests = total_successful = total_unsuccessful = 0.0
    total_applications = total_revenue = 0.0
    total_payment_attempts = 0.0

    for t in tiers:
        successful = min(t.seats_per_day, t.requests_per_day)
        unsuccessful = max(0.0, t.requests_per_day - successful)
        charge = _clip(t.current_surcharge_pct / 100.0 * t.avg_base_fare,
                        t.current_min_charge, t.current_max_charge)
        revenue = successful * charge
        payment_attempts = successful * max(1.0, overhead.lock_release_multiplier)

        tier_results.append(TierResult(
            name=t.name, successful=successful, unsuccessful=unsuccessful,
            applications=t.requests_per_day, revenue=revenue,
        ))

        total_seats += t.seats_per_day
        total_requests += t.requests_per_day
        total_successful += successful
        total_unsuccessful += unsuccessful
        total_applications += t.requests_per_day
        total_revenue += revenue
        total_payment_attempts += payment_attempts

    failed_payment_attempts = max(0.0, total_payment_attempts - total_successful)
    total_overhead = (overhead.current_infra_cost_per_day
                       + failed_payment_attempts * overhead.payment_failure_cost)

    return ScenarioResult(
        name="Current system (FCFS)",
        description="Today's race-to-click Tatkal booking, surcharge paid only by successful bookers.",
        tier_results=tier_results,
        total_seats=total_seats, total_requests=total_requests,
        total_successful=total_successful, total_unsuccessful=total_unsuccessful,
        total_applications=total_applications, total_revenue=total_revenue,
        total_overhead=total_overhead,
    )


def compute_proposed(tiers: List[TierInput], overhead: OverheadAssumptions,
                      demand_multiplier: float = 1.0,
                      scenario_name: str = "Proposed system (computerized random allotment)",
                      scenario_description: str = "") -> ScenarioResult:
    """Model the proposed computerized random allotment system.

    Every applicant pays a flat processing fee. Only successful applicants
    (capped by seat availability, which does NOT change with demand) pay
    the additional reduced premium.

    `demand_multiplier` scales requests/applications only -- capacity
    (seats_per_day) is held fixed, which is exactly the "more people
    apply, same number of seats" scenario the tool is meant to explore.

    Overhead = flat (lower) infrastructure cost + verification cost per
    application (ALL applications, win or lose) + refund/release
    processing cost per unsuccessful applicant.
    """
    tier_results = []
    total_seats = total_requests = total_successful = total_unsuccessful = 0.0
    total_applications = total_revenue = 0.0

    for t in tiers:
        requests = t.requests_per_day * demand_multiplier
        successful = min(t.seats_per_day, requests)
        unsuccessful = max(0.0, requests - successful)
        applications = requests

        flat_fee_revenue = applications * t.proposed_flat_fee
        premium = _clip(t.proposed_premium_pct / 100.0 * t.avg_base_fare,
                         t.proposed_min_premium, t.proposed_max_premium)
        premium_revenue = successful * premium
        revenue = flat_fee_revenue + premium_revenue

        tier_results.append(TierResult(
            name=t.name, successful=successful, unsuccessful=unsuccessful,
            applications=applications, revenue=revenue,
        ))

        total_seats += t.seats_per_day
        total_requests += requests
        total_successful += successful
        total_unsuccessful += unsuccessful
        total_applications += applications
        total_revenue += revenue

    total_overhead = (overhead.proposed_infra_cost_per_day
                       + total_applications * overhead.verification_cost_per_application
                       + total_unsuccessful * overhead.refund_processing_cost_per_unsuccessful)

    if not scenario_description:
        if demand_multiplier == 1.0:
            scenario_description = "Computerized random allotment at today's demand level."
        else:
            scenario_description = (f"Computerized random allotment with demand scaled "
                                     f"{demand_multiplier:g}x versus today (seats unchanged).")

    return ScenarioResult(
        name=scenario_name,
        description=scenario_description,
        tier_results=tier_results,
        total_seats=total_seats, total_requests=total_requests,
        total_successful=total_successful, total_unsuccessful=total_unsuccessful,
        total_applications=total_applications, total_revenue=total_revenue,
        total_overhead=total_overhead,
    )


def run_all_scenarios(tiers: List[TierInput], overhead: OverheadAssumptions,
                       surge_multiplier: float = 3.0) -> Dict[str, ScenarioResult]:
    """Convenience wrapper: run all three standard scenarios at once."""
    current = compute_current(tiers, overhead)
    proposed = compute_proposed(tiers, overhead, demand_multiplier=1.0,
                                 scenario_name="Proposed (same demand)")
    proposed_surge = compute_proposed(
        tiers, overhead, demand_multiplier=surge_multiplier,
        scenario_name=f"Proposed (demand x{surge_multiplier:g})",
    )
    return {
        "current": current,
        "proposed": proposed,
        "proposed_surge": proposed_surge,
    }


# --------------------------------------------------------------------------
# Default illustrative inputs
# --------------------------------------------------------------------------
# NOTE: these are illustrative placeholder figures assembled from public,
# partially outdated sources (see the accompanying report / README) -- NOT
# verified current IRCTC data. Replace with real figures (ideally via an
# RTI reply or an official dataset) before using this tool for anything
# beyond illustration and internal discussion.

def default_tiers() -> List[TierInput]:
    return [
        TierInput(
            name="Non-AC (2S / Sleeper)",
            seats_per_day=90000, requests_per_day=500000, avg_base_fare=400,
            current_surcharge_pct=10, current_min_charge=20, current_max_charge=200,
            proposed_flat_fee=20,
            proposed_premium_pct=5, proposed_min_premium=5, proposed_max_premium=100,
        ),
        TierInput(
            name="Mid-AC (CC / 3AC)",
            seats_per_day=40000, requests_per_day=300000, avg_base_fare=900,
            current_surcharge_pct=30, current_min_charge=125, current_max_charge=225,
            proposed_flat_fee=40,
            proposed_premium_pct=15, proposed_min_premium=65, proposed_max_premium=115,
        ),
        TierInput(
            name="Upper-AC (2AC / Executive)",
            seats_per_day=20000, requests_per_day=150000, avg_base_fare=1500,
            current_surcharge_pct=30, current_min_charge=400, current_max_charge=500,
            proposed_flat_fee=60,
            proposed_premium_pct=15, proposed_min_premium=200, proposed_max_premium=250,
        ),
    ]


def default_overhead() -> OverheadAssumptions:
    return OverheadAssumptions(
        current_infra_cost_per_day=5_000_000,        # INR 50 lakh/day, illustrative
        lock_release_multiplier=2.5,
        payment_failure_cost=15,
        proposed_infra_cost_per_day=1_500_000,        # INR 15 lakh/day, illustrative
        verification_cost_per_application=2,
        refund_processing_cost_per_unsuccessful=1,
    )
