from dotenv import load_dotenv

from chat.txt_extractor import TxtExtractor
from config import ConfigManager

from retrievers.faiss_retriever import PdfFaissRetriever


if __name__ == '__main__':
    pdf_path = ConfigManager().pdf_file_path

    # Add debug prints
    print("Initializing retriever...")
    pdf_retriever = PdfFaissRetriever(pdf_path)

    print("Preparing FAISS index...")
    pdf_retriever.prepare_faiss_index(chunk_size=300)

    print("Running query...")
    query = "retrieve the geographical distribution of revenue"
    similar_chunks = pdf_retriever.retrieve_similar_chunks(query, top_k=3)

    all_chunks = "\n\n---\n\n".join(similar_chunks)

    try:
        load_dotenv()

        extractor = TxtExtractor()
        extraction_result = extractor.extract(all_chunks)
        print(extraction_result)
    except Exception as e:
        print("oops")
        print(f"Error: {str(e)}")
