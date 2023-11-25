# LangChain/Serve/Smith Quick Reference 

## Introduction
Welcome to the comprehensive guide for LangChain, LangServe, and LangSmith. These powerful tools collectively offer a robust framework for building, deploying, and managing advanced AI and language model applications. 

- **LangChain**: A versatile toolkit for creating and managing chains of language models and AI functionalities, facilitating complex tasks and interactions.
- **LangServe**: Dedicated to server-side operations, LangServe manages the deployment and scaling of language models, ensuring efficient and reliable performance.
- **LangSmith**: Focused on tracing, debugging, and detailed analysis, LangSmith provides the necessary tools to monitor, evaluate, and improve AI applications.

This documentation aims to provide users, developers, and AI enthusiasts with a thorough understanding of each tool's capabilities, practical applications, and best practices for integration and usage. Whether you're building sophisticated AI-driven applications or seeking to enhance existing systems with cutting-edge language technologies, this guide will serve as your roadmap to mastering LangChain, LangServe, and LangSmith.

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

---

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

---

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

## Advanced Features

### Section: Embedding Router
- **Objective**: To explain and demonstrate the use of embeddings to dynamically route queries to the most relevant prompt based on semantic similarity. This advanced feature allows LangChain applications to handle a variety of inputs more intelligently.
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

### Section: Managing Prompt Size
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

### Section: Agent Construction and Management
- **Objective**: To demonstrate the process of constructing and managing agents in LangChain. This includes creating agents from runnables and understanding the key components and logic involved in agent operation.
- **Example Code**:
```python
from langchain.agents import AgentExecutor, XMLAgent, tool
from langchain.chat_models import ChatAnthropic

# Initializing the chat model with a specific model version
model = ChatAnthropic(model="claude-2")

# Defining a custom tool for the agent
@tool
def weather_search(query: str) -> str:
    """Tool to search for weather information."""
    # This is a placeholder for actual weather search logic
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
- **Explanation**: This code block illustrates how to build an agent using LangChain's `XMLAgent`. The agent includes a custom tool for weather information and logic to process and format intermediate steps. The agent is executed with a specific query, demonstrating its ability to manage and utilize its components effectively.

---

### Section: Code Writing with LangChain
- **Objective**: To showcase how LangChain can be utilized for writing and executing Python code. This feature enhances the AI's ability to assist in programming tasks, making it a valuable tool for developers.
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

---

### Section: LangServe

#### Basic Deployment and Querying with GPT-3.5-Turbo
- **Example**: Deploying and querying the GPT-3.5-Turbo model using LangServe.
- **Objective**: To illustrate the use of LangServe within the LangChain ecosystem. LangServe is designed to facilitate server-side functionalities for managing and deploying language models, making it an essential tool for scalable and efficient AI applications.
```python
from langserve import LangServeClient

# Initialize the LangServe client
langserve_client = LangServeClient(api_url="https://api.langserve.com")

# Deploying the GPT-3.5-Turbo model
model_config = {
    "model_name": "gpt-3.5-turbo",
    "description": "GPT-3.5 Turbo model for general-purpose use"
}
deployment_response = langserve_client.deploy_model(model_config)
print("Deployment Status:", deployment_response.status)

# Sending a query to the deployed model
query = "Explain the concept of machine learning in simple terms."
response = langserve_client.query_model(model_name="gpt-3.5-turbo", query=query)
print("Model Response:", response.content)
```

#### Advanced Deployment and Custom Configuration
- **Example**: Utilizing LangServe for deploying custom-configured models for specialized tasks.
```python
# Custom deployment with specific parameters
advanced_model_config = {
    "model_name": "custom-gpt-model",
    "description": "A custom-configured GPT model for specialized tasks",
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 150
    }
}
langserve_client.deploy_model(advanced_model_config)

# Querying the custom model
custom_query = "Generate a technical summary of quantum computing."
custom_response = langserve_client.query_model(model_name="custom-gpt-model", query=custom_query)
print("Custom Model Response:", custom_response.content)
```

#### Model Management and Analytics
- **Example**: Managing deployed models and accessing detailed analytics.
```python
# Fetching model analytics
model_analytics = langserve_client.get_model_analytics(model_name="gpt-3.5-turbo")
print("Model Usage Analytics:", model_analytics)

# Updating a deployed model's configuration
update_config = {
    "temperature": 0.5,
    "max_tokens": 200
}
langserve_client.update_model_config(model_name="gpt-3.5-turbo", new_config=update_config)

# Retrieving updated model details
updated_model_details = langserve_client.get_model_details(model_name="gpt-3.5-turbo")
print("Updated Model Details:", updated_model_details)
```

#### Integration with LangChain Applications
- **Example**: Demonstrating seamless integration of LangServe with LangChain.
```python
from langchain.chains import SimpleChain

# Building a SimpleChain with a LangServe deployed model
chain = SimpleChain(model_name="gpt-3.5-turbo", langserve_client=langserve_client)

# Executing the chain with a user query
chain_response = chain.execute("What are the latest trends in AI?")
print("Chain Response using LangServe Model:", chain_response)
```

#### LangSmith Tracing for Enhanced Monitoring
- **Objective**: Showcasing the use of LangSmith tracing within LangServe for detailed monitoring and analysis.
- **Example Code**:
```python
from langserve import LangServeClient
from langsmith import Tracing

# Initialize LangServe client and enable LangSmith tracing
langserve_client = LangServeClient(api_url="https://api.langserve.com")
Tracing.enable()

# Deploying a model with tracing enabled
model_config = {
    "model_name": "gpt-3.5-turbo",
    "description": "GPT-3.5 Turbo model with LangSmith tracing"
}
langserve_client.deploy_model(model_config)

# Query with tracing for detailed interaction logs
query = "Explain the impact of AI on environmental sustainability."
response = langserve_client.query_model(model_name="gpt-3.5-turbo", query=query)
print("Traced Model Response:", response.content)

# Retrieve and analyze trace logs
trace_logs = Tracing.get_logs()
print("Trace Logs:", trace_logs)
```
- **Explanation**: This section highlights the integration of LangSmith tracing in LangServe, enhancing the capability to monitor and analyze model interactions. It is particularly valuable for understanding model behavior, performance optimization, and debugging complex scenarios.

### LangSmith Enhanced Capabilities: Integrating Lilac, Prompt Versioning, and More

#### Introduction
LangSmith, complemented by tools like Lilac, offers advanced capabilities for data analysis and prompt management. This section explores how to leverage these tools for enhanced functionality in LangSmith, incorporating prompt versioning, retrieval QA chains, and editable prompt templates.

#### Integrating Lilac for Enhanced Data Analysis
- **Functionality**: Utilize Lilac to import, enrich, and analyze datasets from LangSmith.
- **Workflow**:
   1. Query datasets from LangSmith.
   2. Import and enrich datasets using Lilac's advanced analysis tools.
   3. Export the processed data for further application within LangSmith.

#### Advanced Prompt Management with Versioning
- **Functionality**: Manage different versions of prompts in LangSmith to ensure consistency and accuracy.
- **Application**:
   1. Track and manage versions of prompts.
   2. Apply specific prompt versions in complex deployments like retrieval QA chains.

#### Retrieval QA Chains
- **Functionality**: Configure retrieval QA chains in LangSmith, leveraging the specific versions of prompts for precise information retrieval.
- **Implementation**:
   1. Define the prompt and its version for the QA chain.
   2. Execute queries using the retrieval QA chain to obtain accurate results.

#### Editable Prompt Templates
- **Functionality**: Use editable prompt templates to customize and experiment with different prompt structures in LangSmith.
- **Usage**:
   1. Create and edit prompt templates dynamically.
   2. Apply edited templates in LangSmith workflows for varied applications.

#### Comprehensive Code Example
```python
# Import necessary libraries
# Import necessary libraries
import langchain
from langchain.prompt_templates import EditablePromptTemplate
# Assuming LangSmith and Lilac libraries are imported correctly

# LangSmith setup (assuming required configurations and authentications are done)
langsmith.initialize(api_key="YOUR_LANGSMITH_API_KEY", endpoint="https://api.langsmith.com")

# Query and fetch datasets from LangSmith using the list_runs method
project_runs = langsmith.client.list_runs(project_name="your_project_name")

# Import dataset into Lilac and enrich it
lilac_dataset = lilac.import_dataset(project_runs)
lilac_dataset.compute_signal(lilac.PIISignal(), 'question')  # Example signal
lilac_dataset.compute_signal(lilac.NearDuplicateSignal(), 'output')  # Another example signal

# Export the enriched dataset for integration with LangSmith
exported_dataset = lilac.export_dataset(lilac_dataset)

# Implementing Prompt Versioning (assuming the existence of such functionality in LangSmith)
prompt_version = 'specific_version_hash'
prompt_name = 'your_prompt_name'
prompt = langsmith.load_prompt(prompt_name, version=prompt_version)

# Configuring a Retrieval QA Chain with the versioned prompt
qa_chain = langchain.RetrievalQAChain(prompt=prompt)

# Execute a query using the QA Chain
query_result = qa_chain.query("What is LangSmith's functionality?")
print(f"QA Chain Query Result: {query_result}")

# Editable Prompt Templates for dynamic prompt editing
editable_prompt = EditablePromptTemplate(prompt_name)
editable_prompt.edit(new_template="New template content for LangSmith")
edited_prompt = editable_prompt.apply()

# Example usage of the edited prompt in a LangSmith application
edited_prompt_result = langsmith.run_prompt(edited_prompt, input_data="Sample input for edited prompt")
print(f"Edited Prompt Result: {edited_prompt_result}")

# Final step: Integrate the exported dataset back into LangSmith for further use
integration_status = langsmith.integrate_dataset(exported_dataset)
if integration_status.success:
    print("Dataset successfully integrated back into LangSmith.")
else:
    print(f"Integration failed with error: {integration_status.error}")
```

#### Conclusion
By integrating these diverse functionalities, LangSmith users can significantly enhance their language model applications. This synergy between LangSmith and tools like Lilac, along with advanced prompt management techniques, paves the way for more sophisticated and effective AI solutions.

---

## Conclusion

In this guide, we have explored the intricate functionalities and applications of LangChain, LangServe, and LangSmith. From building complex AI models with LangChain to deploying and managing them efficiently with LangServe, and ensuring their optimum performance through LangSmith's tracing and debugging, these tools form a comprehensive ecosystem for advanced AI development.

As the field of AI continues to evolve, so will the capabilities and applications of these tools. Please continually explore new features, updates, and best practices to stay ahead in the rapidly advancing world of AI and language models. No document is truly timeless in its teachings, for subsequent wisdom is built upon such. 


For further learning and support, explore the following resources:

- [LangChain Interface](https://python.langchain.com/docs/expression_language/interface)
- [LangChain Cookbook - Prompt + LLM](https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser)
- [LangChain Cookbook - Embedding Router](https://python.langchain.com/docs/expression_language/cookbook/embedding_router)
- [LangChain Cookbook - Agent](https://python.langchain.com/docs/expression_language/cookbook/agent)
- [LangChain Cookbook - Code Writing](https://python.langchain.com/docs/expression_language/cookbook/code_writing)
- [LangChain Cookbook - Memory](https://python.langchain.com/docs/expression_language/cookbook/memory)
- [LangChain Cookbook - Managing Prompt Size](https://python.langchain.com/docs/expression_language/cookbook/prompt_size)
- [LangChain Cookbook - Tools](https://python.langchain.com/docs/expression_language/cookbook/tools)

Thank you for engaging with this documentation. May it be a valuable resource in your journey to mastering LangChain, LangServe, and LangSmith.

---
