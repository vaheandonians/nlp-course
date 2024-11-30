from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from retrievers.faiss_retriever import FaissRetriever


class LocalRagChat:
    def __init__(self, retriever: FaissRetriever):
        # Initialize tokenizer and model separately
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            trust_remote_code=True
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            torch_dtype=torch.float16,
            device_map="mps",
            trust_remote_code=True
        )

        # Create the pipeline
        self.model = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=self.base_model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
        )

        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based on this context:
            {CONTEXT}

            Question: {QUERY}

            Answer:"""
        )
        self.chain = self.prompt | self.model | StrOutputParser()

    def ask(self, query: str):
        context = "\n\n---\n\n".join(self.retriever.retrieve(query))
        return self.chain.invoke({"CONTEXT": context, "QUERY": query})