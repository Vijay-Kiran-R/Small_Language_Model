# eval/domain_eval.py
"""
Domain evaluation for emotion + intent model.
Primary metric: 150–200 hand-crafted test examples (NOT standard benchmarks).
Standard benchmarks do not capture empathy, emotional accuracy, or intent quality.
"""

DOMAIN_TEST_CATEGORIES = {
    'frustration_acknowledgment': 30,   # 30 examples
    'anger_deescalation':         20,
    'urgency_detection':          20,
    'sadness_support':            25,
    'sarcasm_detection':          25,
    'ambiguous_intent':           25,
    'implicit_emotion':           25,
    'technical_frustration':      30,
}
TOTAL_EXAMPLES = sum(DOMAIN_TEST_CATEGORIES.values())  # 200 examples

def evaluate_domain(model, tokenizer, test_examples: list) -> dict:
    """
    Run domain evaluation.
    Each example has: {'input': str, 'expected_behavior': str, 'category': str}
    Score: human rater scores 0/1 per example (or use GPT-4o as judge).
    """
    results = {cat: {'correct': 0, 'total': 0} for cat in DOMAIN_TEST_CATEGORIES}
    for example in test_examples:
        category = example['category']
        # Generate response
        ids = tokenizer.encode(example['input'], add_special_tokens=False)
        # ... (run inference and score against expected_behavior)
        results[category]['total'] += 1
    return results
