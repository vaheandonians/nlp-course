from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import ConfigManager
from retrievers.faiss_retriever import FaissRetriever


class RagChat:

    def __init__(self, retriever: FaissRetriever):
        self.model = ChatOpenAI(model_name=ConfigManager().openai_llm)
        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_template("Answer the question based on this context:\n"
                                                       "{CONTEXT}\n\n\n"
                                                       "Question:\n"
                                                       "{QUERY}")
        self.chain = self.prompt | self.model | StrOutputParser()

    def ask(self, query: str):
        context = "\n\n---\n\n".join(self.retriever.retrieve(query))
        return self.chain.invoke({"CONTEXT": context,"QUERY": query})

#
# if __name__ == "__main__":
#     try:
#         load_dotenv()
#
#         embeddings = OpenAIEmbeddings(model=ConfigManager().openai_embedding_model)
#         index = faiss.IndexFlatL2(len(embeddings.embed_query("hi")))
#
#         vector_store = FAISS(
#             embedding_function=embeddings,
#             index=index,
#             docstore=InMemoryDocstore(),
#             index_to_docstore_id={},
#         )
#
#         loader = PyPDFLoader(ConfigManager().pdf_file_path)
#         docs = loader.load()
#         vector_store.add_documents(docs)
#         result = vector_store.search(PROMPT_TEMPLATE, 'similarity')
#         pprint(result)
#         # pdf = Pdf(ConfigManager().pdf_file_path)
#         # text = pdf.extract_text()
#         # extractor = TxtExtractor(text)
#         # extraction_result = extractor.extract()
#         # print(extraction_result)
#     except Exception as e:
#         print(f"Error: {str(e)}")
