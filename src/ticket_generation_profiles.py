"""
Ticket simulation profiles for drift analysis (see scripts/generate_tickets.py).

Reference profile matches the original assignment-style generator.
Drifted profile increases ticket volume and complaint/negative rates for *non-churn*
customers toward churn-like levels — this shifts feature distributions used in training.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChurnTicketProfile:
    ticket_count_min: int
    ticket_count_max: int
    complaint_prob: float
    negative_prob: float


@dataclass(frozen=True)
class TicketGenerationProfiles:
    churn_yes: ChurnTicketProfile
    churn_no: ChurnTicketProfile


# Original behaviour from generate_tickets.py (reference / production-like baseline)
REFERENCE = TicketGenerationProfiles(
    churn_yes=ChurnTicketProfile(4, 10, 0.5, 0.6),
    churn_no=ChurnTicketProfile(0, 3, 0.1, 0.2),
)

# Simulated operational drift: support load increases; non-churn customers look more
# "angry" in tickets (still with same Churn labels in customers.csv → concept stress).
DRIFTED = TicketGenerationProfiles(
    churn_yes=ChurnTicketProfile(6, 14, 0.55, 0.65),
    churn_no=ChurnTicketProfile(3, 9, 0.35, 0.45),
)

PROFILES: dict[str, TicketGenerationProfiles] = {
    "reference": REFERENCE,
    "drifted": DRIFTED,
}
