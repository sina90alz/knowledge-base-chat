"""Prompt templates for RAG application."""


class PromptTemplates:
    """Collection of prompt templates for various tasks."""

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use the context to provide accurate, detailed answers.
If the context doesn't contain relevant information, say so clearly."""

    RETRIEVAL_PROMPT = """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt."""
        return cls.SYSTEM_PROMPT

    @classmethod
    def get_retrieval_prompt(cls, context: str, question: str) -> str:
        """Get formatted retrieval prompt."""
        return cls.RETRIEVAL_PROMPT.format(context=context, question=question)
