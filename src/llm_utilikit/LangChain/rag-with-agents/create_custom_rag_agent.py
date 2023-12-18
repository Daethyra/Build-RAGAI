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

loader = UnstructuredFileLoader()
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

# Initialize Agent Memory
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
llm = ChatOpenAI(temperature=0)

# Create tools
tool = create_retriever_tool(
    retriever,
    "Search Local Documents",
    "Searches and returns user-supplied documents for contextual generation.",
)
tools = [tool]

system_message = SystemMessage(
    content=(
        "Do your best to answer the questions. "
        "Feel free to use any tools available to look up "
        "relevant information, only if necessary"
    )
)

# This is needed for both the memory and the prompt
memory_key = "history"

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

# Construct the Agent
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)


user_query = input("Please enter your query: ")
result = agent_executor({"input": user_query})
result["output"]