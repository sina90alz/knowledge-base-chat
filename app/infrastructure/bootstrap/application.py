"""Application bootstrap and dependency wiring."""

from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.ports import VectorStore
from app.ingestion.embedder import EmbeddingService
from app.infrastructure.factories import VectorStoreFactory
from app.services.audit_service import AuditService
from app.services.llm import get_llm_service
from app.services.retrieval import RetrievalService
from app.services.verification import AnswerVerificationService


@dataclass(frozen=True)
class ApplicationContainer:
    """Shared application services assembled at startup."""

    embedding_service: EmbeddingService
    vector_store: VectorStore
    retrieval_service: RetrievalService
    llm_service: Any
    verification_service: AnswerVerificationService
    audit_service: AuditService


_application_container: ApplicationContainer | None = None


def create_application() -> ApplicationContainer:
    """Build the application's shared object graph."""
    embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
    embedding_dimension = embedding_service.get_embedding_dimension()
    vector_store = VectorStoreFactory().create_vector_store(embedding_dimension)
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    llm_service = get_llm_service()
    verification_service = AnswerVerificationService(llm_service=llm_service)
    audit_service = AuditService(settings.AUDIT_DB_PATH)

    return ApplicationContainer(
        embedding_service=embedding_service,
        vector_store=vector_store,
        retrieval_service=retrieval_service,
        llm_service=llm_service,
        verification_service=verification_service,
        audit_service=audit_service,
    )


def set_application_container(container: ApplicationContainer) -> None:
    """Set the process-wide application container."""
    global _application_container
    _application_container = container


def get_application_container() -> ApplicationContainer:
    """Return the initialized application container."""
    if _application_container is None:
        raise RuntimeError("Application container has not been initialized")

    return _application_container
