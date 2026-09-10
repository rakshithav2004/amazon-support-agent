INTENTS = {
    "delivery_tracking": {
        "description": "Delivery status, tracking, delayed delivery, missing package, or delivered-but-not-received.",
        "examples": [
            "Where is my package?",
            "My order was supposed to arrive today.",
            "Tracking says delivered but I don't have it."
        ]
    },

    "order_issue": {
        "description": "General order problems that are not primarily delivery, payment, refund, return, or cancellation issues.",
        "examples": [
            "My order disappeared from my order list.",
            "Why is my preorder delayed?",
            "My cart becomes empty when I try to order."
        ]
    },

    "product_issue": {
        "description": "Problems with the received product such as damaged, defective, broken, or wrong item.",
        "examples": [
            "The product arrived damaged.",
            "I received the wrong item.",
            "The item is defective."
        ]
    },

    "return_refund": {
        "description": "Returning an item or receiving/chasing a refund.",
        "examples": [
            "How do I return this item?",
            "Where is my refund?",
            "I returned the product but haven't received my money."
        ]
    },

    "payment_billing": {
        "description": "Payment methods, unexpected charges, billing, invoices, or payment failures.",
        "examples": [
            "Why was I charged?",
            "Which cards can I use?",
            "I cannot pay with my gift card."
        ]
    },

    "account_login": {
        "description": "Account access, login, password, hacked account, or account-security problems.",
        "examples": [
            "I forgot my password.",
            "Someone changed my account email.",
            "I cannot access my account."
        ]
    },

    "prime_membership": {
        "description": "Amazon Prime membership, subscription, Prime benefits, or Prime membership problems.",
        "examples": [
            "I cannot renew my Prime membership.",
            "Why was I charged for Prime?",
            "My Prime membership is not working."
        ]
    },

    "digital_content": {
        "description": "Digital products and services such as Kindle, Prime Video, movies, music, or digital books.",
        "examples": [
            "My Kindle book is not working.",
            "I cannot watch this Prime Video.",
            "Can I return a Kindle book?"
        ]
    },

    "cancellation": {
        "description": "Explicit requests or questions about cancelling an order or service.",
        "examples": [
            "I want to cancel my order.",
            "Can I cancel this order?",
            "I cancelled my order. What happens next?"
        ]
    },

    "other": {
        "description": "Customer support message that does not clearly belong to another intent.",
        "examples": [
            "I need help with something else.",
            "Can someone help me?",
            "I have a question."
        ]
    }
}