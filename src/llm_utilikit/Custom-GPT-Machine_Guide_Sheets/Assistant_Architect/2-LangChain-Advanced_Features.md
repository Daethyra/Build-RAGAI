# LangChain Advanced Features

## Introduction

Building on the core concepts, this guide covers advanced features like embeddings, prompt management, agents, and code writing. These empower sophisticated applications.

## Embedding Router

- Dynamically route queries to relevant prompts using embeddings.

- Key concepts: semantic similarity, embeddings, prompt relevance

```python
from langchain.chat_models import ChatOpenAI    
from langchain.embeddings import OpenAIEmbeddings    
from langchain.prompts import PromptTemplate    
from langchain.schema.output_parser import StrOutputParser    
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough    
from langchain.utils.math import cosine_similarity    

# Creating two distinct prompt templates for different domains   
physics_template = "You are a physics expert. Answer this physics question: {query}"  
math_template = "You are a math expert. Answer this math question: {query}"

# Initializing embeddings and chat model  
embeddings = OpenAIEmbeddings()   
model = ChatOpenAI()  

# Embedding the prompt templates  
prompt_templates = [physics_template, math_template]  
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Defining a function to route the query to the most relevant prompt  
def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    return PromptTemplate.from_template(most_similar)

# Building the chain with embedding-based routing
chain = (   
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router) 
    | model
    | StrOutputParser()  
)

# Example query and response
response = chain.invoke({"query": "What is quantum mechanics?"})
print(response)
```

- This code demonstrates using embeddings and cosine similarity to determine the most relevant prompt template based on the query. It then generates a response using the chosen prompt and chat model.

## Managing Prompt Size

- Strategies for efficiently managing prompt size.

- Key concepts: prompt truncation, content limits   

```python
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_to_openai_function_messages  
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI  
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import WikipediaQueryRun
from langchain.tools.render import format_tool_to_openai_function
from langchain.utilities import WikipediaAPIWrapper

# Installing Wikipedia query package 
# !pip install langchain wikipedia

# Initializing Wikipedia query tool with content character limit 
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=10_000)
)
tools = [wiki]  

# Creating a prompt template with placeholders for user input and agent scratchpad
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}"), 
    MessagesPlaceholder(variable_name="agent_scratchpad"),   
])
llm = ChatOpenAI(model="gpt-3.5-turbo")  

# Building an agent with a focus on managing prompt size
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]  
        ),
    }
    | prompt
    | llm.bind(functions=[format_tool_to_openai_function(t) for t in tools]) 
    | OpenAIFunctionsAgentOutputParser()  
)

# Executing the agent  
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({    
    "input": "What is the tallest mountain?" 
})
print(response)
```

- This code showcases an agent setup focused on managing prompt size by limiting intermediate step content. The query response is generated considering the overall prompt size.

## Agent Construction

- Steps for constructing and managing agents.

- Key concepts: agent components, agent logic

```python  
from langchain.agents import AgentExecutor, XMLAgent, tool
from langchain.chat_models import ChatAnthropic

# Initializing the chat model with a specific model version  
model = ChatAnthropic(model="claude-2")

# Defining a custom tool for the agent 
@tool
def weather_search(query: str) -> str:
    """Tool to search for weather information."""
    # Placeholder for weather search logic   
    return "Sunny with a high of 75 degrees"

tool_list = [weather_search]  

# Retrieving the default prompt for the XMLAgent  
prompt = XMLAgent.get_default_prompt()   

# Defining logic for processing intermediate steps to a string format
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log

# Building an agent from a runnable
agent = (
    {
        "question": lambda x: x["question"],
        "intermediate_steps": lambda x: convert_intermediate_steps(x["intermediate_steps"]), 
    }
    | prompt.partial(tools=lambda: "\n".join([f"{t.name}: {t.description}" for t in tool_list]))
    | model.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgent.get_default_output_parser()   
)

# Executing the agent with a specific query  
agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)
response = agent_executor.invoke({"question": "What's the weather in New York today?"})
print(response)
```

- This code illustrates building an agent in LangChain using `XMLAgent`. It includes a custom tool, logic to process steps, and execution with a specific query.

## Code Writing  

- Utilizing LangChain for writing and executing Python code.

- Key concepts: code generation, PythonREPL

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate   
from langchain.schema.output_parser import StrOutputParser   
from langchain_experimental.utilities import PythonREPL

# Creating a prompt template to instruct the model to write Python code  
template = "Write Python code to solve the following problem: {problem}"
prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{problem}")]) 

# Initializing the chat model
model = ChatOpenAI()

# Function to sanitize and extract Python code from the model's output  
def sanitize_output(text):
    _, after = text.split("```python")
    return after.split("```")[0]

# Building the chain for code writing  
chain = prompt | model | StrOutputParser() | sanitize_output | PythonREPL().run  

# Invoking the chain with a programming problem 
problem = "calculate the factorial of a number"
code_result = chain.invoke({"problem": problem})   
print(code_result)
```

- This code demonstrates using LangChain to automatically generate Python code for a given problem statement. The output is sanitized and executed with PythonREPL.

## Conclusion  

These advanced features enable complex capabilities on top of the core foundations.
