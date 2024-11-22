from pathlib import Path
from typing import List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pypdf  # Upgraded version of PyPDF2

from config import ConfigManager


class PdfFaissRetriever:
    PAGE_SEPARATOR = "\n\n---\n\n"  # Separator for different pages

    def __init__(self, pdf_path: Path, embedding_model: Optional[str] = "paraphrase-MiniLM-L6-v2"):
        self.pdf_path = pdf_path
        self.text = ""
        self.pages = []
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []

        # Validate PDF file path
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")
        if self.pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File {self.pdf_path} is not a PDF")

    def extract_text(self):
        # Load PDF and extract text from each page
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                self.pages = [pdf_reader.pages[page_num].extract_text() for page_num in range(total_pages)]
                self.text = self.PAGE_SEPARATOR.join(self.pages)
            return self.text
        except Exception as e:
            raise Exception(f"Unexpected error while processing PDF: {str(e)}")

    def chunk_text(self, chunk_size: int = 300) -> List[str]:
        # Split the extracted text into chunks of specified chunk_size
        words = self.text.split()
        self.chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return self.chunks

    def prepare_faiss_index(self, chunk_size: int = 300):
        # Extract text, chunk it, and create the FAISS index
        self.extract_text()
        self.chunk_text(chunk_size=chunk_size)

        # Generate embeddings for each chunk
        embeddings = self.model.encode(self.chunks)

        # Create FAISS index and add embeddings
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve_similar_chunks(self, query: str, top_k: int = 5) -> List[str]:
        if not self.index:
            raise ValueError("FAISS index has not been initialized. Call prepare_faiss_index() first.")

        # Generate embedding for the query and search for the most similar chunks
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.chunks[idx] for idx in indices[0]]
        return results


# Usage example
if __name__ == "__main__":
    # Specify the absolute path to your PDF file
    pdf_path = ConfigManager().pdf_file_path

    # Add debug prints
    print("Initializing retriever...")
    pdf_retriever = PdfFaissRetriever(pdf_path)

    print("Preparing FAISS index...")
    pdf_retriever.prepare_faiss_index(chunk_size=300)

    print("Running query...")
    query = "retrieve the geographical distribution of revenue"
    similar_chunks = pdf_retriever.retrieve_similar_chunks(query, top_k=3)
    print(similar_chunks)