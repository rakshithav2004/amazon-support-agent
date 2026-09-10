import json

from src.intents.gemini_classifier import classify
from src.retrieval.retriever import HistoricalRetriever
from src.generation.reply_generator import generate_reply
from src.escalation.decision import decide_escalation


class SupportAgent:

    def __init__(self):
        self.retriever = HistoricalRetriever()

    def handle(self, customer_message: str) -> dict:

        # 1. Classify the customer message.
        intent_result = classify(
            customer_message
        )

        intent = intent_result["intent"]

        # 2. Retrieve similar historical support cases.
        evidence = self.retriever.search(
            customer_message,
            top_k=5
        )

        # 3. Decide whether this can be auto-handled.
        escalation = decide_escalation(
            customer_message,
            intent,
            evidence
        )

        # 4. Generate a grounded draft.
        reply_result = generate_reply(
            customer_message,
            intent,
            evidence
        )

        reply = reply_result["reply"]

        # 5. Make the final customer-facing response
        # consistent with the escalation decision.
        if escalation["decision"] == "escalate":

            reply = (
                "I'm sorry you're dealing with this. "
                "This issue needs to be reviewed by "
                "Amazon Customer Support. Please contact "
                "Customer Support so they can assist you."
            )

        # 6. Return the complete agent result.
        return {
            "customer_message": customer_message,
            "intent": intent,
            "intent_reason": intent_result.get(
                "reason",
                ""
            ),
            "reply": reply,
            "grounding_note": reply_result.get(
                "grounding_note",
                ""
            ),
            "decision": escalation["decision"],
            "decision_reason": escalation["reason"],
            "evidence_ids": [
                item["evidence_id"]
                for item in evidence
            ]
        }


if __name__ == "__main__":

    agent = SupportAgent()

    test_messages = [
        "My package says delivered but I never received it.",

        "Someone hacked my Amazon account and changed my email.",

        "I see an unauthorized charge on my card."
    ]

    for message in test_messages:

        print("\n================================")
        print("CUSTOMER MESSAGE")
        print("================================")

        print(message)

        try:

            result = agent.handle(
                message
            )

            print("\nFINAL AGENT RESPONSE")

            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False
                )
            )

        except Exception as e:

            print(
                f"\nERROR: {e}"
            )