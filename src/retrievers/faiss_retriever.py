from pathlib import Path
from typing import List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pypdf  # Upgraded version of PyPDF2

from artifacts.pdf import Pdf
from config import ConfigManager


class FaissRetriever:
    PAGE_SEPARATOR = "\n\n---\n\n"  # Separator for different pages

    def __init__(self, pdf: Pdf, embedding_model: Optional[str] = "paraphrase-MiniLM-L6-v2"):
        self.pdf = pdf
        self.text = pdf.get_text()
        self.model = SentenceTransformer(embedding_model)
        self.chunks: List[str] = self.chunk_text(self.text)
        self.index = self.prepare_faiss_index(self.chunks, self.model)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
        # Split the extracted text into chunks of specified chunk_size
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    @staticmethod
    def prepare_faiss_index(chunks: List[str], embedding_model):
        # Generate embeddings for each chunk
        embeddings = embedding_model.encode(chunks)

        # Create FAISS index and add embeddings
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        # Generate embedding for the query and search for the most similar chunks
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.chunks[idx] for idx in indices[0]]
        return results
