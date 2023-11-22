# LangChain/Serve/Smith Core Concepts 

## Introduction

This document covers the core concepts needed to get started with LangChain. It aims to provide a focused introduction to the essential building blocks for developing applications with this library.

## Prompt + LLM

- Demonstrates combining a prompt with a language model into a simple LangChain. 

- Key concepts: PromptTemplate, LLM, chain invocation

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

- This code block shows how to create a simple chain that asks the AI to generate a joke based on a user-provided topic. `ChatPromptTemplate` is used to format the prompt, and `ChatOpenAI` is the model that generates the response.

## Memory 

- Illustrates integrating memory for conversational context.

- Key concepts: ConversationBufferMemory, message history

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

- This code demonstrates the use of `ConversationBufferMemory` to keep a record of the conversation. The `ChatPromptTemplate` is configured to include a history of messages, allowing the model to generate responses considering previous interactions.

## Using Tools

- Demonstrates integrating third-party tools into a LangChain application.

- Key concepts: tools, tool integration

```python   
from langchain.chat_models import ChatOpenAI    
from langchain.prompts import ChatPromptTemplate    
from langchain.schema.output_parser import StrOutputParser    
from langchain.tools import DuckDuckGoSearchRun    

# Installing DuckDuckGo search package
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
  
- This example shows using `DuckDuckGoSearchRun` to perform web searches. The user's input is formatted into a search query using `ChatPromptTemplate`, passed through a chat model, and processed by the search tool to retrieve information.

## Conclusion

These core concepts establish the foundation. Continue learning with advanced features and deployment.
```
