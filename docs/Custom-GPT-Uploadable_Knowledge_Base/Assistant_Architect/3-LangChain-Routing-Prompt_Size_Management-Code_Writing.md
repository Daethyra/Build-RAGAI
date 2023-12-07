# LangChain Advanced Features: Embedding Routers, Prompt Size Management, Code Writing

## Introduction

Building on the core concepts, this guide covers advanced features like embeddings, prompt management, agents, and code writing. These empower sophisticated applications.

#  Embedding Router
- **Objective**: To answer "what are routers?" and demonstrate the use of embeddings to dynamically route queries to the most relevant prompt based on semantic similarity. This advanced feature allows LangChain applications to handle a variety of inputs more intelligently.
- **Example Code**:
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
- **Explanation**: This code demonstrates how embeddings and cosine similarity are used to determine which prompt template is most relevant to the user's query. Based on the query's content, it chooses between a physics and a math expert prompt. The response is then generated accordingly by the chat model.

#  Managing Prompt Size
- **Objective**: To illustrate strategies for managing the size of prompts within LangChain applications, ensuring they remain efficient and within the model's context window. This is crucial for maintaining performance, especially in complex chains or agents.
- **Example Code**:
```python
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import WikipediaQueryRun
from langchain.tools.render import format_tool_to_openai_function
from langchain.utilities import WikipediaAPIWrapper

# Installing necessary package for Wikipedia queries
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
- **Explanation**: This code showcases an agent setup that includes a Wikipedia query tool and a prompt template. The agent's construction focuses on managing the prompt size by limiting the content from intermediate steps. The response to a query is generated with consideration to the prompt's overall size, ensuring efficiency.



# Code Writing with LangChain
- **Example Code**:
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
- **Explanation**: This code block demonstrates how LangChain can be used to automatically generate Python code in response to a given problem statement. The `ChatPromptTemplate` guides the AI to focus on code generation, and the output is sanitized and executed using `PythonREPL`. This illustrates LangChain's capability in automating and assisting with coding tasks.