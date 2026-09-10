import re


SECURITY_KEYWORDS = [
    "hacked",
    "hack",
    "stolen account",
    "someone changed my password",
    "someone changed my email",
    "unauthorized access",
    "account compromised",
]

FRAUD_KEYWORDS = [
    "fraud",
    "fraudulent",
    "unauthorized charge",
    "card stolen",
    "someone used my card",
    "charged without permission",
]

HIGH_RISK_INTENTS = {
    "account_login",
    "payment_billing",
}


def contains_keyword(
    message: str,
    keywords: list[str]
) -> bool:

    text = message.lower()

    return any(
        keyword in text
        for keyword in keywords
    )


def decide_escalation(
    customer_message: str,
    intent: str,
    evidence: list[dict]
) -> dict:
    """
    Return whether the case should be auto-handled
    or escalated to a human.
    """

    message = customer_message.lower().strip()

    # --------------------------------------------------
    # Rule 1: Account security
    # --------------------------------------------------

    if contains_keyword(
        message,
        SECURITY_KEYWORDS
    ):
        return {
            "decision": "escalate",
            "reason": (
                "Potential account security compromise "
                "requires human review."
            )
        }

    # --------------------------------------------------
    # Rule 2: Fraud / unauthorized payment
    # --------------------------------------------------

    if contains_keyword(
        message,
        FRAUD_KEYWORDS
    ):
        return {
            "decision": "escalate",
            "reason": (
                "Potential fraud or unauthorized payment "
                "requires human review."
            )
        }

    # --------------------------------------------------
    # Rule 3: High-risk intent with weak evidence
    # --------------------------------------------------

    if intent in HIGH_RISK_INTENTS:

        if not evidence:
            return {
                "decision": "escalate",
                "reason": (
                    "This is a potentially sensitive issue "
                    "and no supporting historical evidence "
                    "was retrieved."
                )
            }

    # --------------------------------------------------
    # Rule 4: No useful historical evidence
    # --------------------------------------------------

    if not evidence:
        return {
            "decision": "escalate",
            "reason": (
                "No relevant historical support evidence "
                "was found, so the system should not guess."
            )
        }

    # --------------------------------------------------
    # Rule 5: Weak retrieval match
    # --------------------------------------------------

    top_score = float(
        evidence[0].get("score", 0)
    )

    if top_score < 0.55:
        return {
            "decision": "escalate",
            "reason": (
                "The best historical match has low "
                "retrieval similarity, so automated "
                "handling may be unreliable."
            )
        }

    # --------------------------------------------------
    # Default: auto-handle
    # --------------------------------------------------

    return {
        "decision": "auto_handle",
        "reason": (
            "A relevant historical support resolution "
            "was found and no high-risk escalation "
            "condition was detected."
        )
    }


if __name__ == "__main__":

    from src.retrieval.retriever import HistoricalRetriever

    retriever = HistoricalRetriever()

    test_cases = [
        {
            "message": (
                "My package says delivered "
                "but I never received it."
            ),
            "intent": "delivery_tracking"
        },
        {
            "message": (
                "Someone hacked my Amazon account "
                "and changed my email."
            ),
            "intent": "account_login"
        },
        {
            "message": (
                "I see an unauthorized charge "
                "on my card."
            ),
            "intent": "payment_billing"
        },
        {
            "message": (
                "I have a completely new problem "
                "that I cannot explain."
            ),
            "intent": "other"
        }
    ]

    print(
        "\n=============================="
    )
    print("ESCALATION TESTS")
    print(
        "=============================="
    )

    for case in test_cases:

        evidence = retriever.search(
            case["message"],
            top_k=5
        )

        decision = decide_escalation(
            case["message"],
            case["intent"],
            evidence
        )

        print("\nCustomer:")
        print(case["message"])

        print("\nDecision:")
        print(decision["decision"])

        print("\nReason:")
        print(decision["reason"])