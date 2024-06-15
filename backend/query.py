from os import environ as env
from functools import lru_cache, partial
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import (
    RetrievalQA as RQA,
    RetrievalQAWithSourcesChain as RQAS,
    create_retrieval_chain, ConversationChain
)
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.config import run_in_executor
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.agents.agent_toolkits.conversational_retrieval.openai_functions import create_conversational_retrieval_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent, create_openai_functions_agent
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import (
    BaseTool, RetrieverInput, _get_relevant_documents, _aget_relevant_documents, tool
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.base import (
    Callbacks,
)
from langchain_core.pydantic_v1 import (
    Field,
)


from typing import List, Any, Optional, Dict
from langchain_core.prompts import ChatPromptTemplate, BasePromptTemplate, PromptTemplate
from pinecone import Pinecone
import requests
import warnings

load_dotenv()

ARXIV_URL = "https://arxiv.org/abs/"

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

SYSTEM_PROMPT = SystemMessage("You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use five sentences maximum and keep the "
    "answer concise."
    "\n\n")



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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt

def custom_prompt():
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    # "Use the following pieces of retrieved context to answer "
    # "the question."
    "If you don't know the answer, say that you "
    "don't know. Use five sentences maximum and keep the "
    "answer concise."
    "\n\n"
    #"{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        #MessagesPlaceholder("chat_history"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    return prompt


def create_agent(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    memory: ConversationBufferWindowMemory,
    system_message: SystemMessage,
    remember_intermediate_steps: bool = True,
    memory_key: str = "chat_history",
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """A convenience method for creating a conversational retrieval agent.

    Args:
        llm: The language model to use, should be ChatOpenAI
        tools: A list of tools the agent has access to
        remember_intermediate_steps: Whether the agent should remember intermediate
            steps or not. Intermediate steps refer to prior action/observation
            pairs from previous questions. The benefit of remembering these is if
            there is relevant information in there, the agent can use it to answer
            follow up questions. The downside is it will take up more tokens.
        memory_key: The name of the memory key in the prompt.
        system_message: The system message to use. By default, a basic one will
            be used.
        verbose: Whether or not the final AgentExecutor should be verbose or not,
            defaults to False.
        max_token_limit: The max number of tokens to keep around in memory.
            Defaults to 2000.

    Returns:
        An agent executor initialized appropriately
    """

    # if remember_intermediate_steps:
    #     memory: BaseMemory = AgentTokenBufferMemory(
    #         memory_key=memory_key, llm=llm, max_token_limit=max_token_limit
    #     )
    # else:
    #     memory = ConversationTokenBufferMemory(
    #         memory_key=memory_key,
    #         return_messages=True,
    #         output_key="output",
    #         llm=llm,
    #         max_token_limit=max_token_limit,
    #     )

    # prompt = OpenAIFunctionsAgent.create_prompt(
    #     system_message=system_message,
    #     extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    # )
    prompt = custom_prompt()
    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        return_intermediate_steps=remember_intermediate_steps,
        **kwargs,
    )



def custom_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
    callbacks: Callbacks = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")
    func = partial(
        _get_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
        callbacks=callbacks
    )
    afunc = partial(
        _aget_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
        callbacks=callbacks
    )
    return Tool(
        name=name,
        description=description,
        func=func,
        coroutine=afunc,
        callbacks=callbacks,
        #config=config,
        tags=tags,
        metadata=metadata,
        args_schema=RetrieverInput,
        verbose=verbose
    )



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
    #streaming=True
)
retriever=docsearch.as_retriever()
#prompt = rag_prompting()


conversational_memory = ConversationSummaryBufferMemory(
    llm = llm,
    max_token_limit = 650,
    memory_key='history',
    return_messages=True,
    input_key='input',
    #output_key='output'
)
# conversational_memory = ConversationBufferWindowMemory(
#     memory_key='history',
#     k=5,
#     return_messages=True,
#     input_key='input',
#     #output_key='output'
# )
# question_answer_chain = create_stuff_documents_chain(
#     llm=llm,
#     #prompt=prompt,
#     memory=conversational_memory
# )
conversation_chain = ConversationChain(
    llm = llm,
    #prompt = prompt,
    memory = conversational_memory,
    verbose=True,
    #return_final_only=False
)

# rag_chain = create_retrieval_chain(
#     retriever,
#     question_answer_chain
# )
rag_chain = create_retrieval_chain(
    retriever,
    conversation_chain
    #return_source_documents=True
)
# retriever_tool = create_retriever_tool(
#     retriever=retriever,
#     name="ArXiv_abstract_searcher",
#     description="use this tool when answering questions to get more information"
# )
retriever_tool = custom_retriever_tool(
    retriever=retriever,
    name="ArXiv_abstract_searcher",
    description="use this tool when answering questions to get more information"
)
# qa = RQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever
# )
def custom_invoke(query):
    output = rag_chain.invoke({"input": query})
    #return ({"output": output})
    return output

@tool
async def custom_ainvoke(query):
    """
    use this tool when answering general queries to get more info about
                a given topic
    """
    return rag_chain.ainvoke({"input": query})
    #return response

tools = [
    Tool(
        name="Abstract_Database",
        #func = rag_chain.invoke,
        func = custom_invoke,
        #func=qa.invoke,
        #coroutine = custom_ainvoke,
        description = ("""
            use this tool when answering general queries to get more info about
                       a given topic
        """)
    )
]

# agent = create_react_agent(
#     #agent='chat-conversational-react-description',
#     tools=tools,
#     llm=llm,
#     prompt=prompt
#     #verbose=True,
#     #max_iterations=3,
#     #early_stopping_method='generate',
#     #memory=conversational_memory
# )
# rag_agent = create_conversational_retrieval_agent(
#     llm=llm,
#     tools=[retriever_tool],
#     #memory=conversational_memory,
#     remember_intermediate_steps=True,
#     verbose=True,
#     max_token_limit=2000
# )
rag_agent = create_agent(
    llm=llm,
    #tools=[retriever_tool],
    tools=tools,
    #tools = [custom_ainvoke],
    memory=conversational_memory,
    memory_key='history',
    system_message=SYSTEM_PROMPT,
    #verbose=True
)
# executor = AgentExecutor(
#     agent = rag_agent,
#     tools = [retriever_tool],
#     #verbose=True,
#     #return_intermediate_steps=True
# )

def connect(query):
    warnings.simplefilter('ignore')
    # async for chunk in rag_agent.astream({"input": query}):
    #     return chunk
    response = rag_agent.invoke({"input": query})
    return response
    # chunks = []
    # async for chunk in rag_agent.astream({"input": query}):
    #     chunks.append(chunk)
    #     print(chunk)
    #     print("---------------------")
    #return chunks
    #yield response
    # for chunk in response:
    #     yield chunk
    # return response
    #return agent({"input": query})
    #response = agent(query)
    #response = rag_chain.invoke({"input": query})]
    #print(response['output'])
    #return response
    # return_dict = {}
    # return_dict['answer'] = response['output']
    # yield return_dict['answer']
    # return_dict['sources'] = response['intermediate_steps']
    # if len(return_dict['sources']) > 1:
    #     for chunk in return_dict['sources']:
    #         yield chunk.metadata
    # return return_dict


    # sources = ""
    # for i in range(len(response['context'])):
    #     source = response['context'][i].metadata
    #     sources += f"{i+1}. [{source['title']}]({ARXIV_URL}{source['source']})  \n  "
    # return response['answer'] \
    #      + "  \n  Sources:  \n  " \
    #      + sources