"""
This module provides an executable script to use an Agent to decide when to search local documents.
It uses UnstructuredFileLoader to load documents, CharacterTextSplitter to split documents into texts,
OpenAIEmbeddings to create embeddings, FAISS to store the embeddings, ChatOpenAI for language model,
AgentTokenBufferMemory for agent memory, OpenAIFunctionsAgent for constructing the agent, and
AgentExecutor for executing
"""

from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.agents import AgentExecutor

# Load documents
loader = UnstructuredFileLoader() # pass in your path as a parameter, here
documents = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Initialize the embeddings module
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) # Ensure your api key is set

# Initialize the vectorstore as `db`
db = FAISS.from_documents(texts, embeddings)

# Initialize our vectorstore as the retriever
retriever = db.as_retriever()

# Initialize Agent Memory
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
llm = ChatOpenAI(temperature=0)

# Create tools
retriever_tool = create_retriever_tool(
    retriever,
    "Search Local Documents",
    "Searches and returns user-supplied documents for contextual generation.",
)
tools = [
    retriever_tool,
    ]

# Here we set a single system message, however it is possible to set multiple role messages in a list
system_message = SystemMessage(
    content=(
        "Do your best to answer the questions. "
        "Feel free to use any tools available to look up "
        "relevant information, only if necessary"
    )
)

# This is needed for both the memory and the prompt
memory_key = "history"

# Create the prompt
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

# Construct the Agent
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

# Construct the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

# Execute the Agent
user_query = input("Please enter your query: ")
result = agent_executor({"input": user_query})
result["output"]
