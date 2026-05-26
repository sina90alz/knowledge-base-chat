"""Answer grounding verification service."""

import json
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
            return self._parse_verification_response(response)

        except Exception as e:
            self.logger.error("Error verifying answer grounding: %s", e)
            raise

    def _parse_verification_response(self, response: str) -> bool:
        """Parse LLM verification response with fallback logic.
        
        Args:
            response: Raw LLM response
            
        Returns:
            True if answer is supported, False otherwise (fail-safe default)
        """
        response_cleaned = response.strip()
        
        # Try pure JSON parsing first
        try:
            data = json.loads(response_cleaned)
            verdict = data.get("verdict", "").strip().upper()
            if verdict == "SUPPORTED":
                self.logger.info("Answer verification: SUPPORTED (JSON)")
                return True
            elif verdict == "UNSUPPORTED":
                self.logger.info("Answer verification: UNSUPPORTED (JSON)")
                return False
        except (json.JSONDecodeError, AttributeError):
            pass

        # Extract JSON object embedded in a chain-of-thought response
        json_match = re.search(
            r'\{[^{}]*"verdict"\s*:\s*"[^"]*"[^{}]*\}', response_cleaned
        )
        if json_match:
            try:
                data = json.loads(json_match.group())
                verdict = data.get("verdict", "").strip().upper()
                if verdict == "SUPPORTED":
                    self.logger.info("Answer verification: SUPPORTED (JSON in CoT)")
                    return True
                elif verdict == "UNSUPPORTED":
                    self.logger.info("Answer verification: UNSUPPORTED (JSON in CoT)")
                    return False
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Fallback: parse as plain text
        normalized_response = response_cleaned.upper()
        
        # Check negative case first (more specific)
        if "UNSUPPORTED" in normalized_response or "NOT SUPPORTED" in normalized_response:
            self.logger.info("Answer verification: UNSUPPORTED (text)")
            return False
        elif re.search(r"\bSUPPORTED\b", normalized_response):
            self.logger.info("Answer verification: SUPPORTED (text)")
            return True
        else:
            # Default to unsupported if unclear (fail-safe)
            self.logger.warning(
                "Unclear verification response, defaulting to UNSUPPORTED: %s",
                response_cleaned[:100]
            )
            return False

    def _build_verification_prompt(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> str:
        """Build a concise grounding verification prompt."""
        return "\n".join(
            [
                "You are a strict grounding checker.",
                "Decide whether the answer is FULLY supported by the provided context.",
                "Only information explicitly stated in the context may be used. Do not rely on general knowledge.",
                "If ANY part of the answer is not found in the context, the verdict must be UNSUPPORTED.",
                "",
                "Step 1: Write ONE sentence explaining your reasoning.",
                'Step 2: On the last line, output ONLY a JSON object: {"verdict": "SUPPORTED"} or {"verdict": "UNSUPPORTED"}.',
                "",
                f"Question: {question}",
                "",
                f"Context: {context}",
                "",
                f"Answer: {answer}",
            ]
        )
