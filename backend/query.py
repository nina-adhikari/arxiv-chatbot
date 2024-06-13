from os import environ as env
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.chains import RetrievalQAWithSourcesChain as RQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.config import run_in_executor
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
import requests

load_dotenv()

# @lru_cache
# def load_embedding(model_name):
#     model_kwargs = {
#         #"device": device
#         }
#     encode_kwargs = {"normalize_embeddings": True, "show_progress_bar": True, "batch_size": 128}
#     hf = HuggingFaceBgeEmbeddings(
#         model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
#     )
#     return hf



def rag_prompting():
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use five sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt

def connect(query):
    model_name = "BAAI/bge-small-en-v1.5"
    #hf = load_embedding(model_name)
    hf_token = env.get("HF_TOKEN")
    pc = Pinecone(api_key=env.get("PINECONE_API_KEY"))
    index = pc.Index("arxiv-abstracts")
    emb = Embeddings(model_id=model_name, hf_token=hf_token)
    #docsearch = PineconeVectorStore(index=index, embedding=hf)
    docsearch = PineconeVectorStore(index=index, embedding=emb)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )
    retriever=docsearch.as_retriever()
    prompt = rag_prompting()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return response



class Embeddings:
    """Interface for embedding models."""
    def __init__(self, model_id, hf_token):
        self.model_id = model_id
        self.hf_token = hf_token

    #@abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """We're not going to bother with implementation since we are just including this for compliance"""
        """Embed search docs."""
        return []

    #@abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.get_embedding_from_hf([text])[0]

    #@lru_cache
    def get_embedding_from_hf(self, texts):
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": texts, "options":{"wait_for_model":True}}
        )
        return response.json()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)