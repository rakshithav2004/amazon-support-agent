from src.escalation.decision import decide_escalation


def test_hacked_account_escalates():

    result = decide_escalation(
        "Someone hacked my account and changed my email.",
        "account_login",
        [
            {
                "score": 0.8
            }
        ]
    )

    assert result["decision"] == "escalate"


def test_unauthorized_payment_escalates():

    result = decide_escalation(
        "I see an unauthorized charge on my card.",
        "payment_billing",
        [
            {
                "score": 0.8
            }
        ]
    )

    assert result["decision"] == "escalate"


def test_normal_delivery_can_auto_handle():

    result = decide_escalation(
        "My package says delivered but I never received it.",
        "delivery_tracking",
        [
            {
                "score": 0.8
            }
        ]
    )

    assert result["decision"] == "auto_handle"


def test_no_evidence_escalates():

    result = decide_escalation(
        "I have a completely new problem.",
        "other",
        []
    )

    assert result["decision"] == "escalate"