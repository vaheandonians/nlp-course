from dotenv import load_dotenv

from artifacts.pdf import Pdf
from chat.rag_chat import RagChat
from config import ConfigManager
from retrievers.faiss_retriever import FaissRetriever

if __name__ == '__main__':
    load_dotenv()

    """
    Pdf: 
        - open the pdf file
        - extract the text from the pdf file
    
    Method:
        - get_text():get the text from the pdf file
    """
    pdf_path = ConfigManager().pdf_file_path
    pdf = Pdf(pdf_path)

    """
    FaissRetriever:
        - invokes get_text to get the text
        - chunks it into Documents
        - creates a FAISS index
        
    Method:
        - retrieve: takes a query and returns the most similar chunks
    """
    retriever = FaissRetriever(pdf)

    """
    RagChat:
        - takes a query through method ask
        - uses the retriever to retrieve the most similar chunks
        - uses the llm to generate a response
    """
    chat = RagChat(retriever)
    answer = chat.ask("What is the geographical distribution of revenue?")

    print(answer)
