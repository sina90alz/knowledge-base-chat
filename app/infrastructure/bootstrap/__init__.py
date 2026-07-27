"""Application composition root."""

from app.infrastructure.bootstrap.application import (
    ApplicationContainer,
    create_application,
    get_application_container,
    set_application_container,
)

__all__ = [
    "ApplicationContainer",
    "create_application",
    "get_application_container",
    "set_application_container",
]
