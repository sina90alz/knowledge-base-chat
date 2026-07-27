"""Application composition root."""

from app.infrastructure.bootstrap.application import (
    ApplicationContainer,
    build_application_container,
    get_application_container,
    set_application_container,
)

__all__ = [
    "ApplicationContainer",
    "build_application_container",
    "get_application_container",
    "set_application_container",
]
