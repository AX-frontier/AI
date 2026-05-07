from __future__ import annotations

import os

import pytest

from agents.main_agent.embedding import DeterministicEmbeddingProvider


def test_deterministic_embedding_supports_query_and_document() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)

    query_vector = provider.embed_query("복수전공 신청 기간")
    document_vector = provider.embed_document("복수전공 신청 안내")

    assert len(query_vector) == 8
    assert len(document_vector) == 8


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="E5 model download/load is an explicit integration test.",
)
def test_e5_embedding_returns_configured_dimensions() -> None:
    from agents.main_agent.embedding import E5EmbeddingProvider

    provider = E5EmbeddingProvider(dimensions=384)
    vector = provider.embed_query("복수전공 신청 기간")

    assert len(vector) == 384
