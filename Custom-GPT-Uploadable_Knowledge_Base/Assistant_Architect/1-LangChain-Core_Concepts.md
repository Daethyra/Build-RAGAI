# LangChain Core Concepts 

## Introduction
Welcome to the comprehensive guide for LangChain, LangServe, and LangSmith.

- **LangChain**: A versatile toolkit for creating and managing chains of language models and AI functionalities, facilitating complex tasks and interactions.
- **LangServe**: Dedicated to server-side operations, LangServe manages the deployment and scaling of language models, ensuring efficient and reliable performance.
- **LangSmith**: Focused on tracing, debugging, and detailed analysis, LangSmith provides the necessary tools to monitor, evaluate, and improve AI applications.

---

## Core Concepts

### Section: Prompt + LLM
- **Objective**: To demonstrate the basic composition of a `PromptTemplate` with a `LLM` (Language Learning Model), creating a chain that takes user input, processes it, and returns the model's output.
- **Example Code**:
```python
from langchain.chat_models import ChatOpenAI   
from langchain.prompts import ChatPromptTemplate   

# Creating a prompt template
prompt = ChatPromptTemplate.from_template("Can you tell me a joke about {topic}?")  

# Initializing the model
model = ChatOpenAI()  

# Building the chain
chain = prompt | model   

# Invoking the chain with user input
response = chain.invoke({"topic": "science"})  
print(response.content)
```
- **Explanation**: This code block shows how to create a simple chain that asks the AI to generate a joke based on a user-provided topic. `ChatPromptTemplate` is used to format the prompt, and `ChatOpenAI` is the model that generates the response.

### Section: Memory
- **Objective**: To illustrate how to integrate memory into a LangChain application, enabling the chain to maintain context across interactions. This is particularly useful for applications like chatbots where retaining context from previous interactions is crucial.
- **Example Code**:
```python
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initializing the chat model
model = ChatOpenAI()

# Creating a prompt template with a placeholder for conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chatbot"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Setting up memory for the conversation
memory = ConversationBufferMemory(return_messages=True)

# Loading initial memory variables
memory.load_memory_variables({})

# Building the chain with memory integration
chain = (
    {"input": "Hello, how are you today?", "history": memory.load_memory_variables()}
    | prompt
    | model
)

# Invoking the chain with user input
response = chain.invoke({"input": "Tell me about LangChain"})
print(response.content)

# Saving the context for future interactions
memory.save_context({"input": "Tell me about LangChain"}, {"output": response.content})
```
- **Explanation**: This code demonstrates the use of `ConversationBufferMemory` to keep a record of the conversation. The `ChatPromptTemplate` is configured to include a history of messages, allowing the model to generate responses considering previous interactions. 

### Section: Using Tools
- **Objective**: To demonstrate how to integrate third-party tools into a LangChain application, thereby enhancing its capabilities. This example will specifically show how to use the `DuckDuckGoSearchRun` tool within a LangChain for web searches.
- **Example Code**:
```python
from langchain.chat_models import ChatOpenAI   
from langchain.prompts import ChatPromptTemplate   
from langchain.schema.output_parser import StrOutputParser   
from langchain.tools import DuckDuckGoSearchRun   

# Installing the necessary package for DuckDuckGo search
# !pip install duckduckgo-search

# Initializing the DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# Creating a prompt template to format user input into a search query
template = "Search for information on: {input}"
prompt = ChatPromptTemplate.from_template(template)

# Initializing the chat model
model = ChatOpenAI()

# Building the chain with search functionality
chain = prompt | model | StrOutputParser() | search

# Invoking the chain with a search query
search_result = chain.invoke({"input": "the latest Python updates"})
print(search_result)
```
- **Explanation**: This example shows the use of `DuckDuckGoSearchRun` to perform web searches. The user's input is formatted into a search query using `ChatPromptTemplate`, passed through a chat model, and then processed by the search tool to retrieve information.

---
