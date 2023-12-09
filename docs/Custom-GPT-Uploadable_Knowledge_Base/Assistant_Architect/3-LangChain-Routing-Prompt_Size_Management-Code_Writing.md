# LangChain Advanced Features: Embedding Routers, Prompt Size Management, Code Writing

##  Embedding Router
Routing allows you to create non-deterministic chains where the output of a previous step defines the next step. Routing helps provide structure and consistency around interactions with LLMs.

As a very simple example, let’s suppose we have two templates optimized for different types of questions, and we want to choose the template based on the user input.

```python
from langchain.prompts import PromptTemplate

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""
physics_prompt = PromptTemplate.from_template(physics_template)

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""
math_prompt = PromptTemplate.from_template(math_template)
```

### Using LangChain Expression Language (LCEL)
We can easily do this using a RunnableBranch. A RunnableBranch is initialized with a list of (condition, runnable) pairs and a default runnable. It selects which branch by passing each condition the input it’s invoked with. It selects the first condition to evaluate to True, and runs the corresponding runnable to that condition with the input.

If no provided conditions match, it runs the default runnable.

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
general_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question as accurately as you can.\n\n{input}"
)
prompt_branch = RunnableBranch(
    (lambda x: x["topic"] == "math", math_prompt),
    (lambda x: x["topic"] == "physics", physics_prompt),
    general_prompt,
)

from typing import Literal

from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel
from langchain.utils.openai_functions import convert_pydantic_to_openai_function


class TopicClassifier(BaseModel):
    "Classify the topic of the user question"

    topic: Literal["math", "physics", "general"]
    "The topic of the user question. One of 'math', 'physics' or 'general'."


classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
llm = ChatOpenAI().bind(
    functions=[classifier_function], function_call={"name": "TopicClassifier"}
)
parser = PydanticAttrOutputFunctionsParser(
    pydantic_schema=TopicClassifier, attr_name="topic"
)
classifier_chain = llm | parser

from operator import itemgetter

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

final_chain = (
    RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain)
    | prompt_branch
    | ChatOpenAI()
    | StrOutputParser()
)

final_chain.invoke(
    {
        "input": "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3?"
    }
)
```

- **Embedding Routing: End-to-end example**:
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

---

#  Managing Prompt Size
Agents dynamically call tools. The results of those tool calls are added back to the prompt, so that the agent can plan the next action. Depending on what tools are being used and how they’re being called, the agent prompt can easily grow larger than the model context window.

With LCEL, it’s easy to add custom functionality for managing the size of prompts within your chain or agent. Let’s look at simple agent example that can search Wikipedia for information.

- **Manage Prompt Size: End-to-end Example**:
```python
# Installing necessary package for Wikipedia queries
# !pip install langchain wikipedia

from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import ChatPromptValue
from langchain.tools import WikipediaQueryRun
from langchain.tools.render import format_tool_to_openai_function
from langchain.utilities import WikipediaAPIWrapper

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
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

Let’s try a many-step question without any prompt size handling:

# Create a function to condense the prompt for the sake of our context window
def condense_prompt(prompt: ChatPromptValue) -> ChatPromptValue:
    messages = prompt.to_messages()
    num_tokens = llm.get_num_tokens_from_messages(messages)
    ai_function_messages = messages[2:]
    while num_tokens > 4_000:
        ai_function_messages = ai_function_messages[2:]
        num_tokens = llm.get_num_tokens_from_messages(
            messages[:2] + ai_function_messages
        )
    messages = messages[:2] + ai_function_messages
    return ChatPromptValue(messages=messages)

agent = (
    {
        "input": itemgetter("input"),
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | condense_prompt
    | llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke(
    {
        "input": "Who is the current US president? What's their home state? What's their home state's bird? What's that bird's scientific name?"
    }
)
```

---

# Code Writing using LangChain's `PythonREPL`
- **How it works**: This code block demonstrates how LangChain can be used to automatically generate Python code in response to a given problem statement. The `ChatPromptTemplate` guides the AI to focus on code generation, and the output is sanitized and executed using `PythonREPL`. This illustrates LangChain's capability in automating and assisting with coding tasks.

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
