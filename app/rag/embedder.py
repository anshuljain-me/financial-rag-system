import time
from typing import List
from google import genai
from google.genai import types
from app.core.config import get_settings

settings = get_settings()

class FinancialEmbedder:
    """
    Financial Embedding Service leveraging Google Gemini text-embedding models
    with Matryoshka Representation Learning (MRL) configured to 768 dimensions.
    """

    MODEL_NAME = "gemini-embedding-2"

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY.strip().strip("'").strip('"'))
        self.output_dimensionality = settings.EMBEDDING_DIMENSION

    def embed_text(self, text: str) -> List[float]:
        """Generates a 768-dimensional dense vector for document chunks."""
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return [0.0] * self.output_dimensionality

        for attempt in range(3):
            try:
                response = self.client.models.embed_content(
                    model=self.MODEL_NAME,
                    contents=clean_text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.output_dimensionality,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                return list(response.embeddings[0].values)
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2.0)
                else:
                    break

        return [0.0] * self.output_dimensionality

    def embed_query(self, query: str) -> List[float]:
        """Generates a 768-dimensional dense vector optimized for query retrieval."""
        clean_query = query.replace("\n", " ").strip()
        if not clean_query:
            return [0.0] * self.output_dimensionality

        for attempt in range(3):
            try:
                response = self.client.models.embed_content(
                    model=self.MODEL_NAME,
                    contents=clean_query,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.output_dimensionality,
                        task_type="RETRIEVAL_QUERY"
                    )
                )
                return list(response.embeddings[0].values)
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2.0)
                else:
                    break

        return [0.0] * self.output_dimensionality

# Backward compatibility aliases
GeminiEmbedder = FinancialEmbedder
EmbeddingService = FinancialEmbedder
