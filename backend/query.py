# region Basic imports

from dataclasses import dataclass, field
from dotenv import load_dotenv
# from functools import lru_cache, partial
from os import environ as env
from typing import Any, Dict, List, Optional
import requests
import warnings

# endregion

# region langchain imports

from langchain.agents import (
    AgentExecutor,
    Tool,
    # create_react_agent
)
from langchain.agents.openai_functions_agent.base import (
    # OpenAIFunctionsAgent,
    create_openai_functions_agent
)
from langchain.chains import (
    ConversationChain,
    # RetrievalQA as RQA,
    # RetrievalQAWithSourcesChain as RQAS,
    create_retrieval_chain
)
from langchain.chains.base import Chain
from langchain.chains.conversation.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory
)

# endregion

# region langchain_core imports
from langchain_core.callbacks.base import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.memory import BaseMemory
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    # PromptTemplate
)
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import (
    BaseTool,
    # RetrieverInput,
    # _aget_relevant_documents,
    # _get_relevant_documents,
    # tool
)

# endregion

# region Other Langchain imports

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

# endregion

# Pinecone (vector database)

from pinecone import Pinecone

# region Unused imports

# from langchain.agents.agent_toolkits.conversational_retrieval.openai_functions import \
#     create_conversational_retrieval_agent
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# endregion

MODEL_NAME = "BAAI/bge-small-en-v1.5"
HF_TOKEN_KEY = "HF_TOKEN"
PINECONE_KEY = "PINECONE_API_KEY"
OPENAI_KEY = "OPENAI_API_KEY"
INDEX_NAME = "arxiv-abstracts"

@dataclass
class Embeddings:
    """Interface for embedding models."""
    model_id: str
    hf_token: str

    #@abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs. We're not going to bother with implementation since we are just including this for compliance."""
        return []

    #@abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.get_embedding_from_hf([text])[0]

    #@lru_cache
    def get_embedding_from_hf(self, texts: List[str]) -> Any:
        """Get the embedding of texts from the model hosted on Hugging Face."""
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


@dataclass
class ConversationAgentWrapper:
    llm: BaseLanguageModel
    retriever: BaseRetriever
    prompt: BasePromptTemplate
    memory_key: str = "history"
    input_key: str = "input"
    memory: BaseMemory = field(init=False)
    chain: Chain = field(init=False)
    tools: List[BaseTool] = field(init=False)
    agent: Runnable = field(init=False)
    executor: AgentExecutor = field(init=False)

    def __post_init__(self):
        self.memory = self.create_memory()
        self.chain = self.create_chain()
        self.tools = self.create_tools()
        self.agent = self.create_agent()
        self.executor = self.create_executor()

    def create_memory(self) -> BaseMemory:
        return ConversationSummaryBufferMemory(
            llm = self.llm,
            max_token_limit = 650,
            memory_key=self.memory_key,
            return_messages=True,
            input_key=self.input_key,
            #output_key='output'
        )
    
    def create_chain(self) -> Chain:
        conversation_chain = ConversationChain(
        llm = self.llm,
        #prompt = prompt,
        memory = self.memory,
        #verbose=True,
        #return_final_only=False
        )

        return create_retrieval_chain(
            self.retriever,
            conversation_chain
        )
    
    def invoke_chain(self, query: str) -> Any:
        print("Chain invoked")
        return self.chain.invoke({self.input_key: query})

    def create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="Abstract_Database",
                func = self.invoke_chain,
                #coroutine = custom_ainvoke,
                description = ("""
                    use this tool when answering questions to get more information about a given topic
                """)
            )
        ]

    def create_agent(self) -> Runnable:
        """Create and return a conversation retrieval agent.
        
        Returns:
            An agent initialized appropriately
        """
        return create_openai_functions_agent(
            llm = self.llm,
            tools = self.tools,
            prompt = self.prompt
        )
    
    def create_executor(self,
        remember_intermediate_steps: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create and return a conversational retrieval agent executor. Assumes an agent already exists.

        Args:
            remember_intermediate_steps: Whether the agent should remember intermediate
                steps or not. Intermediate steps refer to prior action/observation
                pairs from previous questions. The benefit of remembering these is if
                there is relevant information in there, the agent can use it to answer
                follow up questions. The downside is it will take up more tokens.
            verbose: Whether or not the final AgentExecutor should be verbose or not,
                defaults to False.

        Returns:
            An agent executor initialized appropriately
        """
        return AgentExecutor(
            agent = self.agent,
            tools = self.tools,
            memory = self.memory,
            verbose = verbose,
            return_intermediate_steps =remember_intermediate_steps,
            **kwargs,
        )
    
    def invoke_executor(self, input):
        return self.executor.invoke({self.input_key: input})


def create_prompt(memory_key: str = "history"):
    system_prompt = (
    "You are an assistant for question-answering tasks. You also have"
    "access to a tool to help you gather more information."
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
        MessagesPlaceholder(memory_key),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    return prompt

def create_retriever(
        hft_key: str = HF_TOKEN_KEY,
        pc_key: str = PINECONE_KEY,
        model_name: str = MODEL_NAME,
        index_name: str = INDEX_NAME
):
    """
    Load the vector store and return it as a retriever.
    """
    # Load environment variables
    load_dotenv()
    hf_token = env.get(hft_key)
    pinecone_key = env.get(pc_key)
    
    pc = Pinecone(api_key = pinecone_key)
    index = pc.Index(index_name)
    embedding = Embeddings(model_id = model_name, hf_token = hf_token)
    vector_store = PineconeVectorStore(index = index, embedding = embedding)
    return vector_store.as_retriever()

def setup():
    prompt = create_prompt()
    retriever = create_retriever()
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        #streaming=True
    )
    
    return ConversationAgentWrapper(
        llm = llm,
        retriever = retriever,
        prompt = prompt,
    )

agent = setup()

def connect(query):
    warnings.simplefilter('ignore')
    response = agent.invoke_executor(query)
    return response



# Unused stuff

# region Unused memories

# conversational_memory = ConversationBufferWindowMemory(
#     memory_key='history',
#     k=5,
#     return_messages=True,
#     input_key='input',
#     #output_key='output'
# )

# endregion

# region Unused prompts

# def rag_prompting():
#     system_prompt = (
#         "You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. If you don't know the answer, say that you "
#         "don't know. Use five sentences maximum and keep the "
#         "answer concise."
#         "\n\n"
#         "{context}"
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ])
#     return prompt

# endregion

# region Unused chains

# question_answer_chain = create_stuff_documents_chain(
#     llm=llm,
#     #prompt=prompt,
#     memory=conversational_memory
# )

# rag_chain = create_retrieval_chain(
#     retriever,
#     question_answer_chain
# )

# qa = RQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever
# )

# endregion

# region Unused tools

# retriever_tool = create_retriever_tool(
#     retriever=retriever,
#     name="ArXiv_abstract_searcher",
#     description="use this tool when answering questions to get more information"
# )

# retriever_tool = custom_retriever_tool(
#         retriever=retriever,
#         name="ArXiv_abstract_searcher",
#         description="use this tool when answering questions to get more information"
#     )

# @tool
# async def custom_ainvoke(query):
#     """
#     use this tool when answering general queries to get more info about
#                 a given topic
#     """
#     return rag_chain.ainvoke({"input": query})

# def custom_retriever_tool(
#     retriever: BaseRetriever,
#     name: str,
#     description: str,
#     *,
#     document_prompt: Optional[BasePromptTemplate] = None,
#     document_separator: str = "\n\n",
#     callbacks: Callbacks = None,
#     tags: Optional[List[str]] = None,
#     metadata: Optional[Dict[str, Any]] = None,
#     verbose: bool = True,
#     **kwargs: Any,
# ) -> Tool:
#     """Create a tool to do retrieval of documents.

#     Args:
#         retriever: The retriever to use for the retrieval
#         name: The name for the tool. This will be passed to the language model,
#             so should be unique and somewhat descriptive.
#         description: The description for the tool. This will be passed to the language
#             model, so should be descriptive.

#     Returns:
#         Tool class to pass to an agent
#     """
#     document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")
#     func = partial(
#         _get_relevant_documents,
#         retriever=retriever,
#         document_prompt=document_prompt,
#         document_separator=document_separator,
#         callbacks=callbacks
#     )
#     afunc = partial(
#         _aget_relevant_documents,
#         retriever=retriever,
#         document_prompt=document_prompt,
#         document_separator=document_separator,
#         callbacks=callbacks
#     )
#     return Tool(
#         name=name,
#         description=description,
#         func=func,
#         coroutine=afunc,
#         callbacks=callbacks,
#         #config=config,
#         tags=tags,
#         metadata=metadata,
#         args_schema=RetrieverInput,
#         verbose=verbose
#     )

# endregion

# region Unused agents

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

# endregion

# region Unused functions

# def connect(query):
#     warnings.simplefilter('ignore')
#     # async for chunk in rag_agent.astream({"input": query}):
#     #     return chunk
#     response = rag_agent.invoke({"input": query})
#     return response
#     # chunks = []
#     # async for chunk in rag_agent.astream({"input": query}):
#     #     chunks.append(chunk)
#     #     print(chunk)
#     #     print("---------------------")
#     #return chunks
#     #yield response
#     # for chunk in response:
#     #     yield chunk
#     # return response
#     #return agent({"input": query})
#     #response = agent(query)
#     #response = rag_chain.invoke({"input": query})]
#     #print(response['output'])
#     #return response
#     # return_dict = {}
#     # return_dict['answer'] = response['output']
#     # yield return_dict['answer']
#     # return_dict['sources'] = response['intermediate_steps']
#     # if len(return_dict['sources']) > 1:
#     #     for chunk in return_dict['sources']:
#     #         yield chunk.metadata
#     # return return_dict


#     # sources = ""
#     # for i in range(len(response['context'])):
#     #     source = response['context'][i].metadata
#     #     sources += f"{i+1}. [{source['title']}]({ARXIV_URL}{source['source']})  \n  "
#     # return response['answer'] \
#     #      + "  \n  Sources:  \n  " \
#     #      + sources

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

# endregion