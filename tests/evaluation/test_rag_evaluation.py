from tests.evaluation.rag_cases import RAG_EVALUATION_CASES


def test_rag_evaluation_cases_are_well_formed() -> None:
    assert len(RAG_EVALUATION_CASES) >= 3

    for case in RAG_EVALUATION_CASES:
        assert case["name"]
        assert case["query"]
        assert case["expected_source"]