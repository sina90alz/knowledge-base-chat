"""Answer grounding verification service."""

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMGenerationService(Protocol):
    """Protocol for services that generate text from prompts."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


class AnswerVerificationService:
    """Verify that generated answers are grounded in provided context."""

    def __init__(self, llm_service: LLMGenerationService) -> None:
        """Initialize answer verification service.

        Args:
            llm_service: Service used to generate verification responses
        """
        self.llm_service = llm_service
        self.logger = logger
        self.logger.info("AnswerVerificationService initialized")

    def verify_answer(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> bool:
        """Return True when the answer is fully supported by the context."""
        prompt = self._build_verification_prompt(
            question=question,
            context=context,
            answer=answer,
        )

        try:
            response = self.llm_service.generate(prompt)
            normalized_response = response.strip().upper()
            is_supported = bool(
                re.search(r"\bSUPPORTED\b", normalized_response)
            ) and "UNSUPPORTED" not in normalized_response

            self.logger.info("Answer verification result: %s", normalized_response)
            return is_supported

        except Exception as e:
            self.logger.error("Error verifying answer grounding: %s", e)
            raise

    def _build_verification_prompt(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> str:
        """Build a concise grounding verification prompt."""
        return "\n".join(
            [
                "Decide whether the answer is fully supported by the context.",
                "Respond ONLY with SUPPORTED or UNSUPPORTED.",
                "",
                f"Question: {question}",
                "",
                f"Context: {context}",
                "",
                f"Answer: {answer}",
            ]
        )
