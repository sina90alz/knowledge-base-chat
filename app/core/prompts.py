"""Prompt templates for RAG application."""

from app.core.config import settings


class PromptTemplates:
    """Collection of prompt templates for various tasks."""

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use the context to provide accurate, detailed answers.
If the context doesn't contain relevant information, say so clearly."""

    RETRIEVAL_PROMPT = """Answer the question using ONLY the information in the context below.

STRICT RULES:
- Use ONLY facts explicitly stated in the context
- Do NOT use external knowledge or make assumptions
- If information is missing or unclear, say: "I cannot answer this based on the provided context."
- Do NOT speculate or infer beyond what is written

Context:
{context}

Question: {question}

Answer:"""

    OLLAMA_RETRIEVAL_PROMPT = """You are a helpful assistant answering a question from retrieved documents.

STRICT RULES:
- Use ONLY the context below - do NOT use external knowledge
- If the answer is not in the context, say: "I don't know based on the available documents."
- Do NOT make assumptions or speculate
- Quote or reference specific parts when possible
- Keep the answer clear and concise

Context:
{context}

Question:
{question}

Answer:"""

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt."""
        return cls.SYSTEM_PROMPT

    @classmethod
    def get_retrieval_prompt(cls, context: str, question: str) -> str:
        """Get formatted retrieval prompt."""
        if settings.LLM_PROVIDER.strip().lower() == "ollama":
            return cls.OLLAMA_RETRIEVAL_PROMPT.format(
                context=context,
                question=question,
            )

        return cls.RETRIEVAL_PROMPT.format(context=context, question=question)
