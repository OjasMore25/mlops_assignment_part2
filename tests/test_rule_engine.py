from src.rule_engine import compute_risk


def test_high_risk_many_tickets():
    '''
        This test checks whether the system classifies a customer as HIGH risk when:
        1. The contract type is month-to-month
        2. The number of tickets in the past 30 days is greater than 5
        3. The number of complaint tickets is 0

        According to the rule engine:
        tickets_last_30_days > 5 → HIGH risk

        Therefore, this instance must be classified as HIGH.
    '''

    row = {
        "contract_type": "Month-to-month",
        "tickets_last_30_days": 7,
        "complaint_ticket": 0
    }

    assert compute_risk(row) == "HIGH"


def test_high_risk_complaint_monthly_contract():
    '''
        This test checks whether the system classifies a customer as HIGH risk when:
        1. The contract type is month-to-month
        2. The customer has at least one complaint ticket
        3. The number of tickets in the past 30 days is less than the threshold for rule 1

        According to the rule engine:
        month-to-month contract + complaint ticket → HIGH risk

        Therefore, this instance must be classified as HIGH.
    '''

    row = {
        "contract_type": "Month-to-month",
        "tickets_last_30_days": 2,
        "complaint_ticket": 1
    }

    assert compute_risk(row) == "HIGH"


def test_medium_risk():
    '''
        This test checks whether the system classifies a customer as MEDIUM risk when:
        1. The contract type is not month-to-month
        2. The number of tickets in the past 30 days is greater than or equal to 3
        3. No complaint tickets are present

        According to the rule engine:
        tickets_last_30_days >= 3 → MEDIUM risk
        (provided higher priority rules are not triggered)

        Therefore, this instance must be classified as MEDIUM.
    '''

    row = {
        "contract_type": "Two year",
        "tickets_last_30_days": 3,
        "complaint_ticket": 0
    }

    assert compute_risk(row) == "MEDIUM"


def test_low_risk():
    '''
        This test checks whether the system classifies a customer as LOW risk when:
        1. The contract type is not month-to-month
        2. The number of tickets in the past 30 days is less than 3
        3. No complaint tickets are present

        Since none of the HIGH or MEDIUM risk rules are triggered,
        the system should classify the customer as LOW risk.
    '''

    row = {
        "contract_type": "Two year",
        "tickets_last_30_days": 1,
        "complaint_ticket": 0
    }

    assert compute_risk(row) == "LOW"


def test_high_rule_boundary():
    '''
        This test checks the boundary condition for the rule:
        tickets_last_30_days > 5 → HIGH risk

        Here the number of tickets is exactly 5, which does NOT satisfy the condition (>5).

        Since:
        1. The contract type is not month-to-month
        2. Complaint tickets are 0
        3. tickets_last_30_days = 5

        The system should fall back to the MEDIUM risk rule
        (tickets_last_30_days >= 3).

        Therefore, this instance must be classified as MEDIUM.
    '''

    row = {
        "contract_type": "Two year",
        "tickets_last_30_days": 5,
        "complaint_ticket": 0
    }

    assert compute_risk(row) == "MEDIUM"


def test_rule_precedence():
    '''
        This test verifies rule precedence in the rule engine.

        The customer satisfies two conditions:
        1. tickets_last_30_days >= 3 → MEDIUM risk
        2. month-to-month contract + complaint ticket → HIGH risk

        Since HIGH risk rules have higher priority,
        the system must classify this instance as HIGH.

        This test ensures the rule engine correctly prioritizes rules.
    '''

    row = {
        "contract_type": "Month-to-month",
        "tickets_last_30_days": 3,
        "complaint_ticket": 1
    }

    assert compute_risk(row) == "HIGH"


def test_complaint_not_monthly():
    '''
        This test verifies that a complaint ticket alone does not trigger HIGH risk
        if the contract type is not month-to-month.

        In this case:
        1. Contract type is "Two year"
        2. Complaint tickets = 1
        3. tickets_last_30_days = 2

        Since the HIGH risk rule requires BOTH:
        month-to-month contract AND complaint ticket,

        and the MEDIUM risk rule requires:
        tickets_last_30_days >= 3,

        none of the rules are triggered.

        Therefore, the system must classify this instance as LOW risk.
    '''

    row = {
        "contract_type": "Two year",
        "tickets_last_30_days": 2,
        "complaint_ticket": 1
    }

    assert compute_risk(row) == "LOW"