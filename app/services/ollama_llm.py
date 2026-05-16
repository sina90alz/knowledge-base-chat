"""Ollama LLM service for local answer generation."""

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OllamaLLMService:
    """Generate text with a local Ollama model."""

    def __init__(
        self,
        model_name: str = "tinyllama",
        base_url: str = "http://localhost:11434/api/generate",
    ) -> None:
        """Initialize the Ollama service.

        Args:
            model_name: Name of the local Ollama model to use
            base_url: Ollama generate API endpoint
        """
        self.model_name = model_name
        self.base_url = base_url
        logger.info("OllamaLLMService initialized with model: %s", model_name)

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using Ollama.

        Args:
            prompt: Prompt to send to the local model

        Returns:
            Generated text only

        Raises:
            ValueError: If prompt is empty
            requests.RequestException: If the Ollama API request fails
            KeyError: If the Ollama response does not contain generated text
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()
            generated_text = data["response"]
            return str(generated_text)

        except requests.RequestException as e:
            logger.error("Error calling Ollama API: %s", e)
            raise
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Invalid Ollama response: %s", e)
            raise
