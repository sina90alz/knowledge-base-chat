"""LLM service for answer generation."""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMService:
    """Generate text with an OpenAI-compatible chat model."""

    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        """Initialize the LLM client.

        Args:
            model_name: Name of the chat model to use

        Raises:
            ValueError: If OPENAI_API_KEY is not configured
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        logger.info("LLMService initialized with model: %s", model_name)

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Prompt to send to the model

        Returns:
            Generated text only

        Raises:
            ValueError: If prompt is empty
            Exception: If generation fails
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            generated_text = response.choices[0].message.content
            return generated_text or ""

        except Exception as e:
            logger.error("Error generating LLM response: %s", e)
            raise
