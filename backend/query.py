# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from os import environ as env
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.chains import RetrievalQAWithSourcesChain as RQA
from functools import lru_cache
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate




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
    hf = load_embedding("BAAI/bge-small-en-v1.5")
    pc = Pinecone(api_key=env.get("PINECONE_API_KEY"))
    index = pc.Index("arxiv-abstracts")
    docsearch = PineconeVectorStore(index=index, embedding=hf)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )
    retriever=docsearch.as_retriever()
    qa_with_sources = RQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    prompt = rag_prompting()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return response
    #answer = qa_with_sources(query)
    #return answer