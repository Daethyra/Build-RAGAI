# LangChain Expression Language (LCEL)
## LLMs VS Chat Models
LLMs and chat models are subtly but importantly different. LLMs in LangChain refer to pure text completion models. The APIs they wrap take a string prompt as input and output a string completion. OpenAI's GPT-3 is implemented as an LLM. Chat models are often backed by LLMs but tuned specifically for having conversations. And, crucially, their provider APIs use a different interface than pure text completion models. Instead of a single string, they take a list of chat messages as input. Usually these messages are labeled with the speaker (usually one of "System", "AI", and "Human"). And they return an AI chat message as output. GPT-4 and Anthropic's Claude-2 are both implemented as chat models.

### Prompts vs PromptTemplate
A prompt for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output, such as answering questions, completing sentences, or engaging in a conversation.

## Concepts
1. Build Python code to prompt an LLM
2. Memory Integration
3. Retrieval Augmented Generation
4. Agents

## 1. Prompt + LLM
The most common and valuable composition is taking:
`PromptTemplate` / `ChatPromptTemplate` -> `LLM` / `ChatModel` -> `OutputParser`
- Almost any other chains you build will use this building block.

### - **PromptTemplate + LLM**: 
The simplest composition is just combining a prompt and model to create a chain that takes user input, adds it to a prompt, passes it to a model, and returns the raw model output.

Note, you can mix and match PromptTemplate/ChatPromptTemplates and LLMs/ChatModels as you like here.
```python
from langchain.chat_models import ChatOpenAI   
from langchain.prompts import ChatPromptTemplate   

# Creating a prompt template
prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}?")  
model = ChatOpenAI()  
chain = prompt | model   

# Invoking the chain with user input
response = chain.invoke({"topic": "science"})  
print(response.content)
```
#### `PromptTemplate`
Use `PromptTemplate` to create a template for a string prompt.

By default, `PromptTemplate` uses Python’s str.format syntax for templating.
```
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_template.format(adjective="funny", content="chickens")
```
`'Tell me a funny joke about chickens.'`
The template supports any number of variables, including no variables:
```
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke")
prompt_template.format()
```
`'Tell me a joke'`
##### Adding Validation
For additional validation, specify input_variables explicitly. These variables will be compared against the variables present in the template string during instantiation, raising an exception if there is a mismatch. For example:
```
from langchain.prompts import PromptTemplate

invalid_prompt = PromptTemplate(
    input_variables=["adjective"],
    template="Tell me a {adjective} joke about {content}.",
)
```
```
ValidationError: 1 validation error for PromptTemplate
__root__
  Invalid prompt schema; check for mismatched or missing input parameters. 'content' (type=value_error)
  ```

#### `ChatPromptTemplate`
The prompt to chat models is a list of chat messages.

Each chat message is associated with content, and an additional parameter called `role`. For example, in the OpenAI Chat Completions API, a chat message can be associated with an AI assistant, a human or a system role.

Create a chat prompt template like this:
```
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
```

`ChatPromptTemplate.from_messages` accepts a variety of message representations.

For example, in addition to using the 2-tuple representation of (type, content) used above, you could pass in an instance of `MessagePromptTemplate` or `BaseMessage`.
```
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

llm = ChatOpenAI()
llm(chat_template.format_messages(text="i dont like eating tasty things."))
```
`AIMessage(content='I absolutely love indulging in delicious treats!')`
This provides you with a lot of flexibility in how you construct your chat prompts.

#### Attaching Function Call information
- Create custom, re-useable functions

```
functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]
chain = prompt | model.bind(function_call={"name": "joke"}, functions=functions)
```
`chain.invoke({"foo": "bears"}, config={})`
Response:  `AIMessage(content='', additional_kwargs={'function_call': {'name': 'joke', 'arguments': '{\n  "setup": "Why don\'t bears wear shoes?",\n  "punchline": "Because they have bear feet!"\n}'}}, example=False)`

### PromptTemplate + LLM + OutputParser
We can also add in an output parser to easily transform the raw LLM/ChatModel output into a more workable format.
```
from langchain.schema.output_parser import StrOutputParser

chain = prompt | model | StrOutputParser()
```
Notice that this now returns a string - a much more workable format for downstream tasks
`chain.invoke({"foo": "bears"})`
Response: `"Why don't bears wear shoes?\n\nBecause they have bear feet!"`

#### Functions Output Parser
When you specify the function to retun, you must just want to parse that directly
```
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonOutputFunctionsParser()
)
```
```chain.invoke({"foo": "bears"})```
```
{'setup': "Why don't bears like fast food?",
 'punchline': "Because they can't catch it!"}
 ```

 ```
 from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)
```

```chain.invoke({"foo": "bears"})```

```"Why don't bears wear shoes?"```

### Simplifying Input
To make invocation even simpler, we can add a `RunnableParallel` to take care of creating the prompt input dict for us:
```
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

map_ = RunnableParallel(foo=RunnablePassthrough())
chain = (
    map_
    | prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)
```

Since we’re composing our map with another Runnable, we can even use some syntactic sugar and just use a dict:
```
chain = (
    {"foo": RunnablePassthrough()}
    | prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)
```

In either composition, we still invoke like this:
`chain.invoke("bears")`
`"Why don't bears like fast food?"`

### Chat Models
Chat models are a variation on language models. While chat models use language models under the hood, the interface they use is a bit different. Rather than using a “text in, text out” API, they use an interface where “chat messages” are the inputs and outputs.
#### Messages
The chat model interface is based around messages rather than raw text. The types of messages currently supported in LangChain are `AIMessage`, `HumanMessage`, `SystemMessage`, `FunctionMessage` and `ChatMessage` – `ChatMessage` takes in an arbitrary role parameter. Most of the time, you’ll just be dealing with HumanMessage, AIMessage, and `SystemMessage`
#### LangChain Expression Language (LCEL)
Chat models implement the [Runnable interface](https://python.langchain.com/docs/expression_language/interface "Online documentation"), the basic building block of the LangChain Expression Language (LCEL). This means they support `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`, `astream_log` calls.

Chat models accept `List[BaseMessage]` as inputs, or objects which can be coerced to messages, including `str` (converted to `HumanMessage`) and `PromptValue`.
```
from langchain.schema.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]
```
```
chat.invoke(messages)
```
```
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```
```
chat.batch([messages])
```
```
await chat.ainvoke(messages)
```
```
async for chunk in chat.astream(messages):
    print(chunk.content, end="", flush=True)
```
```
async for chunk in chat.astream_log(messages):
    print(chunk)
```

---

## 2. Memory
### **Adding Memory**: 
This shows how to add memory to an arbitrary chain. Right now, you can use the memory classes but need to hook it up manually
```python
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
```
`memory = ConversationBufferMemory(return_messages=True)`
`memory.load_memory_variables({})`
`{'history': []}`
```
chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
)
```
```
inputs = {"input": "hi im bob"}
response = chain.invoke(inputs)
memory.save_context(inputs, {"output": response.content})
memory.load_memory_variables({})
```

```
{'history': [HumanMessage(content='hi im bob', additional_kwargs={}, example=False),
  AIMessage(content='Hello Bob! How can I assist you today?', additional_kwargs={}, example=False)]}
```
```
inputs = {"input": "whats my name"}
response = chain.invoke(inputs)
response
```
`AIMessage(content='Your name is Bob.', additional_kwargs={}, example=False)`

This code demonstrates the use of `ConversationBufferMemory` to keep a record of the conversation. The `ChatPromptTemplate` is configured to include a history of messages, allowing the model to generate responses considering previous interactions.

#### Conversation Buffer End2End Example
Finally, let's take a look at using this in a chain. We'll use an `LLMChain`, and show working with both an LLM and a ChatModel.

**Using an LLM**
```
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


llm = OpenAI(temperature=0)
# Notice that "chat_history" is present in the prompt template
template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""
prompt = PromptTemplate.from_template(template)
# Notice that we need to align the `memory_key`
memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
conversation({"question": "hi"})
```
**Using a ChatModel**
```
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


llm = ChatOpenAI()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
conversation({"question": "hi"})
```

### Chat Messages | Quickest implementation
One of the core utility classes underpinning most (if not all) memory modules is the `ChatMessageHistory` class. This is a super lightweight wrapper that provides convenience methods for saving HumanMessages, AIMessages, and then fetching them all.

You may want to use this class directly if you are managing memory outside of a chain.

```
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")

history.messages
```

### Vector store-backed retriever
A vector store retriever is a retriever that uses a vector store to retrieve documents. It is a lightweight wrapper around the vector store class to make it conform to the retriever interface. It uses the search methods implemented by a vector store, like similarity search and MMR, to query the texts in the vector store.

Once you construct a vector store, it's very easy to construct a retriever. Let's walk through an example.
```
from langchain.document_loaders import TextLoader
loader = TextLoader('../../../state_of_the_union.txt')

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```
#### Maximum Marginal Relevance Retrieval
By default, the vector store retriever uses similarity search. If the underlying vector store supports maximum marginal relevance search, you can specify that as the search type.
```
retriever = db.as_retriever(search_type="mmr")
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```
#### Similarity Score Threshold Retrieval
You can also set a retrieval method that sets a similarity score threshold and only returns documents with a score above that threshold.
```
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```
#### Specifying top_k
```
retriever = db.as_retriever(search_kwargs={"k": 1})
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```

### Backed by a Vector Store
`VectorStoreRetrieverMemory` stores memories in a vector store and queries the top-K most "salient" docs every time it is called.

This differs from most of the other Memory classes in that it doesn't explicitly track the order of interactions.

In this case, the "docs" are previous conversation snippets. This can be useful to refer to relevant pieces of information that the AI was told earlier in the conversation.

```
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
```

#### Initialize Your Vector Store
Depending on the store you choose, this step may look different. Consult the relevant vector store documentation for more details.
```
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS


embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
```
#### Create your `VectorStoreRetrieverMemory`
The memory object is instantiated from any vector store retriever.
```
# In actual usage, you would set `k` to be a higher value, but we use k=1 to show that
# the vector lookup still returns the semantically relevant information
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# When added to an agent, the memory object can save pertinent information from conversations or used tools
memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})
memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"})
# Notice the first result returned is the memory pertaining to tax help, which the language model deems more semantically relevant
# to a 1099 than the other documents, despite them both containing numbers.
print(memory.load_memory_variables({"prompt": "what sport should i watch?"})["history"])
```
### Using in a Chain
Let's walk through an example, again setting `verbose=True` so we can see the prompt.
```
llm = OpenAI(temperature=0) # Can be any valid LLM
_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    # We set a very low max_token_limit for the purposes of testing.
    memory=memory,
    verbose=True
)
conversation_with_summary.predict(input="Hi, my name is Perry, what's up?")
# Here, the basketball related content is surfaced
conversation_with_summary.predict(input="what's my favorite sport?")
# Even though the language model is stateless, since relevant memory is fetched, it can "reason" about the time.
# Timestamping memories and data is useful in general to let the agent determine temporal relevance
conversation_with_summary.predict(input="Whats my favorite food")
# The memories from the conversation are automatically stored,
# since this query best matches the introduction chat above,
# the agent is able to 'remember' the user's name.
conversation_with_summary.predict(input="What's my name?")
```

#### LangSmith Tracing
All `ChatModel`s come with built-in LangSmith tracing. Just set the following environment variables:

```
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY=<your-api-key>
```
and any ChatModel invocation (whether it’s nested in a chain or not) will automatically be traced. A trace will include inputs, outputs, latency, token usage, invocation params, environment params, and more. See an example [here](https://smith.langchain.com/public/a54192ae-dd5c-4f7a-88d1-daa1eaba1af7/r).

In LangSmith you can then provide feedback for any trace, compile annotated datasets for evals, debug performance in the playground, and more.

---

### 3. Retrieval Augmented Generation



---

### 4. Agents 
#### Construction and Management
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