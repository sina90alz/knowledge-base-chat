"""FastAPI dependency providers for application services."""

from typing import Any

from fastapi import Request

from app.services.audit_service import AuditService
from app.services.retrieval import RetrievalService
from app.services.verification import AnswerVerificationService


def get_container(request: Request) -> Any:
    """Return the application container stored on FastAPI state."""
    return request.app.state.container


def get_retrieval_service(request: Request) -> RetrievalService:
    """Return the startup-wired retrieval service."""
    return get_container(request).retrieval_service


def get_llm_service(request: Request) -> Any:
    """Return the startup-wired LLM service."""
    return get_container(request).llm_service


def get_verification_service(request: Request) -> AnswerVerificationService:
    """Return the startup-wired answer verification service."""
    return get_container(request).verification_service


def get_audit_service(request: Request) -> AuditService:
    """Return the startup-wired audit service."""
    return get_container(request).audit_service
