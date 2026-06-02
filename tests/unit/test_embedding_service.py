import numpy as np

from app.ingestion.embedder import EmbeddingService


def test_embed_text_uses_injected_model(fake_embedding_model) -> None:
    service = EmbeddingService(model=fake_embedding_model)

    embedding = service.embed_text("hello")

    np.testing.assert_array_equal(
        embedding,
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    assert embedding.dtype == np.float32
    assert fake_embedding_model.calls == [
        ("hello", {"convert_to_numpy": True}),
    ]


def test_constructor_does_not_load_provider_until_first_use(
    fake_model_provider,
) -> None:
    service = EmbeddingService(model_provider=fake_model_provider)

    assert fake_model_provider.load_count == 0
    assert service.get_embedding_dimension() == 3
    assert fake_model_provider.load_count == 1
