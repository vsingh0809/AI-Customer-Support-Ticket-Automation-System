"""Deterministic RAG evaluation cases."""


RAG_EVALUATION_CASES = [
    {
        "name": "refund_policy",
        "query": "What is the refund policy?",
        "expected_source": "refund_policy.md",
    },
    {
        "name": "shipping_policy",
        "query": "How long does shipping take?",
        "expected_source": "shipping_policy.md",
    },
    {
        "name": "payment_policy",
        "query": "Which payment methods are supported?",
        "expected_source": "payment_policy.md",
    },
]