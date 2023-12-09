# LangChain Expression Language (LCEL)
## Critical-Comprehension: Fundamental Concepts

### Interface
To make it as easy as possible to create custom chains, we've implemented a "Runnable" protocol. The Runnable protocol is implemented for most components. This is a standard interface, which makes it easy to define custom chains as well as invoke them in a standard way. The standard interface includes:

- `stream`: stream back chunks of the response
```python
for s in chain.stream({"topic": "bears"}):
    print(s.content, end="", flush=True)
```
- `invoke`: call the chain on an input
```python
chain.invoke({"topic": "bears"})
```
- `batch`: call the chain on a list of inputs
```python
# You can set the number of concurrent requests by using the max_concurrency parameter
chain.batch([{"topic": "bears"}, {"topic": "cats"}], config={"max_concurrency": 5})
```

These also have corresponding async methods:

- `astream`: stream back chunks of the response async
```python
async for s in chain.astream({"topic": "bears"}):
    print(s.content, end="", flush=True)
```
- `ainvoke`: call the chain on an input async
```python
async for s in chain.astream({"topic": "bears"}):
    print(s.content, end="", flush=True)
```
- `abatch`: call the chain on a list of inputs async
```python
await chain.abatch([{"topic": "bears"}, {"topic": "cats"}], config={"max_concurrency": 5})
```
- `astream_log`: All runnables also have a method .astream_log() which is used to stream (as they happen) all or part of the intermediate steps of your chain/sequence. This is useful to show progress to the user, to use intermediate results, or to debug your chain. [Learn More](https://python.langchain.com/docs/expression_language/interface "LangChain Documentation")

The input type and output type varies by component:

| Component    | Input Type                                         | Output Type        |
|--------------|----------------------------------------------------|--------------------|
| Prompt       | Dictionary                                         | PromptValue        |
| ChatModel    | Single string, list of chat messages or a PromptValue | ChatMessage       |
| LLM          | Single string, list of chat messages or a PromptValue | String            |
| OutputParser | The output of an LLM or ChatModel                  | Depends on the parser |
| Retriever    | Single string                                      | List of Documents  |
| Tool         | Single string or dictionary, depending on the tool | Depends on the tool |

All runnables expose input and output schemas to inspect the inputs and outputs:
- `input_schema`: an input Pydantic model auto-generated from the structure of the Runnable
- `output_schema`: an output Pydantic model auto-generated from the structure of the Runnable

Let's take a look at these methods. To do so, we'll create a super simple PromptTemplate + ChatModel chain.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model
```

**Input Schema**:
This is a Pydantic model dynamically generated from the structure of any Runnable. You can call .schema() on it to obtain a JSONSchema representation.
```python
# The input schema of the chain is the input schema of its first part, the prompt.
chain.input_schema.schema()
prompt.input_schema.schema()
model.input_schema.schema()
```

**Output Schema**:
This is a Pydantic model dynamically generated from the structure of any Runnable. You can call .schema() on it to obtain a JSONSchema representation.
```python
# The output schema of the chain is the output schema of its last part, in this case a ChatModel, which outputs a ChatMessage
chain.output_schema.schema()
```

### LLMs VS Chat Models
LLMs and chat models are subtly but importantly different. LLMs in LangChain refer to pure text completion models. The APIs they wrap take a string prompt as input and output a string completion. OpenAI's GPT-3 is implemented as an LLM. Chat models are often backed by LLMs but tuned specifically for having conversations. And, crucially, their provider APIs use a different interface than pure text completion models. Instead of a single string, they take a list of chat messages as input. Usually these messages are labeled with the speaker (usually one of "System", "AI", and "Human"). And they return an AI chat message as output. GPT-4 and Anthropic's Claude-2 are both implemented as chat models.

### Prompts vs PromptTemplate
A prompt for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output, such as answering questions, completing sentences, or engaging in a conversation.

---

## Concepts
1. Build Python code to prompt an LLM
2. Memory Integration
3. Retrieval Augmented Generation
4. Agents and Tools

# 1. Prompt + LLM
The most common and valuable composition is taking:
`PromptTemplate` / `ChatPromptTemplate` -> `LLM` / `ChatModel` -> `OutputParser`
- Almost any other chains you build will use this building block.

## - Invoking "LLM"s: PromptTemplate + LLM 
The simplest composition is just combining a prompt and model to create a chain that takes user input, adds it to a prompt, passes it to a model, and returns the raw model output.

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

The most common type of chaining in any LLM application is combining a prompt template with an LLM and optionally an output parser.

`BasePromptTemplate`, `BaseLanguageModel` and `BaseOutputParser` all implement the `Runnable` interface and are designed to be piped into one another, making LCEL composition very easy:
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

prompt = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)
runnable = prompt | ChatOpenAI() | StrOutputParser()
runnable.invoke({"product": "colorful socks"})
```

### `PromptTemplate`
Use `PromptTemplate` to create a template for a string prompt.

By default, `PromptTemplate` uses Python’s str.format syntax for templating.
```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_template.format(adjective="funny", content="chickens")
```
`'Tell me a funny joke about chickens.'`
The template supports any number of variables, including no variables:
```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke")
prompt_template.format()
```
`'Tell me a joke'`

#### Adding Validation
For additional validation, specify input_variables explicitly. These variables will be compared against the variables present in the template string during instantiation, raising an exception if there is a mismatch. For example:
```
from langchain.prompts import PromptTemplate

invalid_prompt = PromptTemplate(
    input_variables=["adjective"],
    template="Tell me a {adjective} joke about {content}.",
)
```

```shell
ValidationError: 1 validation error for PromptTemplate
__root__
  Invalid prompt schema; check for mismatched or missing input parameters. 'content' (type=value_error)
```

### `ChatPromptTemplate`
The prompt to chat models is a list of chat messages.

Each chat message is associated with content, and an additional parameter called `role`. For example, in the OpenAI Chat Completions API, a chat message can be associated with an AI assistant, a human or a system role.

Create a chat prompt template like this:
```python
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
```python
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

### Attaching Function Call information
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

## PromptTemplate + LLM + OutputParser
We can also add in an output parser to easily transform the raw LLM/ChatModel output into a more workable format.

```python
from langchain.schema.output_parser import StrOutputParser

chain = prompt | model | StrOutputParser()
```

Notice that this now returns a string - a much more workable format for downstream tasks
`chain.invoke({"foo": "bears"})`
Response: `"Why don't bears wear shoes?\n\nBecause they have bear feet!"`

### Functions Output Parser
When you specify the function to retun, you must parse that directly

```python
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonOutputFunctionsParser()
)
```
```chain.invoke({"foo": "bears"})```
```shell
{'setup': "Why don't bears like fast food?",
 'punchline': "Because they can't catch it!"}
```

```python
 from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)
chain.invoke({"foo": "bears"})
```
- Why don't bears wear shoes"?

## `chain.invoke()`: Simplifying Input with `RunnableParallel`
To make invocation even simpler, we can add a `RunnableParallel` to take care of creating the prompt input dict for us:
```python
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
```python
chain = (
    {"foo": RunnablePassthrough()}
    | prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)
```
In either composition, we still invoke like this:
```python
chain.invoke("bears")
# Response would be: "Why don't bears like fast food?"
```

## Invoking Chat Models
Chat models are a variation on language models. While chat models use language models under the hood, the interface they use is a bit different. Rather than using a “text in, text out” API, they use an interface where “chat messages” are the inputs and outputs.
### Messages
The chat model interface is based around messages rather than raw text. The types of messages currently supported in LangChain are `AIMessage`, `HumanMessage`, `SystemMessage`, `FunctionMessage` and `ChatMessage` – `ChatMessage` takes in an arbitrary role parameter. Most of the time, you’ll just be dealing with HumanMessage, AIMessage, and `SystemMessage`
### LangChain Expression Language (LCEL)
Chat models implement the [Runnable interface](https://python.langchain.com/docs/expression_language/interface "Online documentation"), the basic building block of the LangChain Expression Language (LCEL). This means they support `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`, `astream_log` calls.

Chat models accept `List[BaseMessage]` as inputs, or objects which can be coerced to messages, including `str` (converted to `HumanMessage`) and `PromptValue`.
```python
from langchain.schema.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]

chat.invoke(messages)
hunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)

chat.batch([messages])

await chat.ainvoke(messages)

async for chunk in chat.astream(messages):
    print(chunk.content, end="", flush=True)

async for chunk in chat.astream_log(messages):
    print(chunk)
```

---

# 2. Memory
## **Adding Memory**: 
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

### Splitting Text: Character Splitting, Split by Tokens,
- The default recommended text splitter is the RecursiveCharacterTextSplitter. This text splitter takes a list of characters. It tries to create chunks based on splitting on the first character, but if any chunks are too large it then moves onto the next character, and so forth. By default the characters it tries to split on are ["\n\n", "\n", " ", ""]

#### Recursive Character Splitting

- In addition to controlling which characters you can split on, you can also control a few other things:

* `length_function`: how the length of chunks is calculated. Defaults to just counting number of characters, but it's pretty common to pass a token counter here.
* `chunk_size`: the maximum size of your chunks (as measured by the length function).
* `chunk_overlap`: the maximum overlap between chunks. It can be nice to have some overlap to maintain some continuity between chunks (e.g. do a sliding window).
* `add_start_index`: whether to include the starting position of each chunk within the original document in the metadata.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)
```




### Conversation Buffer End2End Example
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

## Chat Messages | Quickest implementation
One of the core utility classes underpinning most (if not all) memory modules is the `ChatMessageHistory` class. This is a super lightweight wrapper that provides convenience methods for saving HumanMessages, AIMessages, and then fetching them all.

You may want to use this class directly if you are managing memory outside of a chain.

```
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")

history.messages
```

## Vector store-backed retriever
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
### Maximum Marginal Relevance Retrieval
By default, the vector store retriever uses similarity search. If the underlying vector store supports maximum marginal relevance search, you can specify that as the search type.
```
retriever = db.as_retriever(search_type="mmr")
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```
### Similarity Score Threshold Retrieval
You can also set a retrieval method that sets a similarity score threshold and only returns documents with a score above that threshold.
```
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```
### Specifying top_k
```
retriever = db.as_retriever(search_kwargs={"k": 1})
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```

## Backed by a Vector Store
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

### Initialize Your Vector Store
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
### Create your `VectorStoreRetrieverMemory`
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
## Using in a Chain
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

### LangSmith Tracing
All `ChatModel`s come with built-in LangSmith tracing. Just set the following environment variables:

```
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY=<your-api-key>
```
and any ChatModel invocation (whether it’s nested in a chain or not) will automatically be traced. A trace will include inputs, outputs, latency, token usage, invocation params, environment params, and more. See an example [here](https://smith.langchain.com/public/a54192ae-dd5c-4f7a-88d1-daa1eaba1af7/r).

In LangSmith you can then provide feedback for any trace, compile annotated datasets for evals, debug performance in the playground, and more.

---

# 3. Retrieval Augmented Generation (RAG)
## Example 1: Basic RAG
Let’s look at adding in a retrieval step to a prompt and LLM, which adds up to a “retrieval-augmented generation” chain
### Dependencies
`pip install langchain openai faiss-cpu tiktoken`
```
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
chain.invoke("where did harrison work?")
```
```
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)
chain.invoke({"question": "where did harrison work", "language": "italian"})
```

## Example 2: Conversational Retrieval Chain
We can easily add in conversation history. This primarily means adding in chat_message_history

```
from langchain.schema import format_document
from langchain.schema.messages import get_buffer_string
from langchain.schema.runnable import RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage

from langchain.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}
# Prepare qa_chain
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
# Invoke qa_chain
conversational_qa_chain.invoke(
    {
        "question": "where did harrison work?",
        "chat_history": [],
    }
)
```
```
conversational_qa_chain.invoke(
    {
        "question": "where did he work?",
        "chat_history": [
            HumanMessage(content="Who wrote this notebook?"),
            AIMessage(content="Harrison"),
        ],
    }
)
```

## Example 3: Add Memory and Returning Source Documents
This shows how to use memory with the above. For memory, we need to manage that outside at the memory. For returning the retrieved documents, we just need to pass them through all the way.
```
from operator import itemgetter
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

# Invoke using the `final_chain`
inputs = {"question": "where did harrison work?"}
result = final_chain.invoke(inputs)
result
```
```
# Note that the memory does not save automatically
# This will be improved in the future
# For now you need to save it yourself
memory.save_context(inputs, {"answer": result["answer"].content})

# Load memory
memory.load_memory_variables({})

# Further questioning
inputs = {"question": "but where did he really work?"}
result = final_chain.invoke(inputs)
result
```


## Example !X!: Setup
### Dependencies
We’ll use an OpenAI chat model and embeddings and a Chroma vector store in this walkthrough, but everything shown here works with any [ChatModel](https://python.langchain.com/docs/integrations/chat/) or [LLM](https://python.langchain.com/docs/integrations/llms/), [Embeddings](https://python.langchain.com/docs/integrations/text_embedding/), and [VectorStore](https://python.langchain.com/docs/integrations/vectorstores/) or [Retriever](https://python.langchain.com/docs/integrations/retrievers).

We’ll use the following packages:
`pip install -U langchain openai chromadb langchainhub bs4`

### RAG Quickstart
```python
import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")

# cleanup
vectorstore.delete_collection()
```
#### Adding Sources
With LCEL it’s easy to return the retrieved documents or certain source metadata from the documents:
```
from operator import itemgetter

from langchain.schema.runnable import RunnableParallel

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableParallel(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

rag_chain_with_source.invoke("What is Task Decomposition")
```

#### Adding Memory
Suppose we want to create a stateful application that remembers past user inputs. There are two main things we need to do to support this. 1. Add a messages placeholder to our chain which allows us to pass in historical messages 2. Add a chain that takes the latest user query and reformulates it in the context of the chat history into a standalone question that can be passed to our retriever.

Let’s start with 2. We can build a “condense question” chain that looks something like this:
```
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

condense_q_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
condense_q_chain = condense_q_prompt | llm | StrOutputParser()

from langchain.schema.messages import AIMessage, HumanMessage

condense_q_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model"),
        ],
        "question": "What is meant by large",
    }
)
```

```python
condense_q_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model"),
        ],
        "question": "How do transformers work",
    }
)
```

And now we can build our full QA chain. Notice we add some routing functionality to only run the “condense question chain” when our chat history isn’t empty.

```python
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]


rag_chain = (
    RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
    | qa_prompt
    | llm
)

chat_history = []

question = "What is Task Decomposition?"
ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg])

second_question = "What are common ways of doing it?"
rag_chain.invoke({"question": second_question, "chat_history": chat_history})
```

## Add Message History (memory)
The `RunnableWithMessageHistory` let’s us add message history to certain types of chains.

Specifically, it can be used for any Runnable that takes as input one of * a sequence of `BaseMessage` a dict with a key that takes a sequence of `BaseMessage` a dict with a key that takes the latest message(s) as a string or sequence of `BaseMessage`, and a separate key that takes historical messages

And returns as output one of a string that can be treated as the contents of an `AIMessage` a sequence of `BaseMessage` * a dict with a key that contains a sequence of `BaseMessage`

Let’s take a look at some examples to see how it works.

### Setup
We’ll use Redis to store our chat message histories and Anthropic’s claude-2 model so we’ll need to install the following dependencies:
`pip install -U langchain redis anthropic`

Set your Anthropic API key:
```
import getpass
import os

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
```
Start a local Redis Stack server if we don’t have an existing Redis deployment to connect to:
`docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest`
`REDIS_URL = "redis://localhost:6379/0"`

#### Example: Messages Input, Dictionary Output
Let’s create a simple chain that takes a dict as input and returns a `BaseMessage`.

In this case the `"question"` key in the input represents our input message, and the `"history"` key is where our historical messages will be injected.
```
from typing import Optional

from langchain.chat_models import ChatAnthropic
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.chat_history import BaseChatMessageHistory
from langchain.schema.runnable.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant who's good at {ability}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatAnthropic(model="claude-2")
```
### Adding Message History
To add message history to our original chain we wrap it in the `RunnableWithMessageHistory` class.

Crucially, we also need to define a method that takes a session_id string and based on it returns a `BaseChatMessageHistory`. Given the same input, this method should return an equivalent output.

In this case we’ll also want to specify `input_messages_key` (the key to be treated as the latest input message) and `history_messages_key` (the key to add historical messages to).
```
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
    input_messages_key="question",
    history_messages_key="history",
)
```

### Invoking with Configuration
Whenever we call our chain with message history, we need to include a config that contains the `session_id`
`config={"configurable": {"session_id": "<SESSION_ID>"}}`
Given the same configuration, our chain should be pulling from the same chat message history.
```
chain_with_history.invoke(
    {"ability": "math", "question": "What does cosine mean?"},
    config={"configurable": {"session_id": "foobar"}},
)
```
#### Example: Messages Input, Dictionary Output
```python
from langchain.schema.messages import HumanMessage
from langchain.schema.runnable import RunnableParallel

chain = RunnableParallel({"output_message": ChatAnthropic(model="claude-2")})
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
    output_messages_key="output_message",
)

chain_with_history.invoke(
    [HumanMessage(content="What did Simone de Beauvoir believe about free will")],
    config={"configurable": {"session_id": "baz"}},
)
```
AI Response: 
`{'output_message': AIMessage(content=' Here is a summary of Simone de Beauvoir\'s views on free will:\n\n- De Beauvoir was an existentialist philosopher and believed strongly in the concept of free will. She rejected the idea that human nature or instincts determine behavior.\n\n- Instead, de Beauvoir argued that human beings define their own essence or nature through their actions and choices. As she famously wrote, "One is not born, but rather becomes, a woman."\n\n- De Beauvoir believed that while individuals are situated in certain cultural contexts and social conditions, they still have agency and the ability to transcend these situations. Freedom comes from choosing one\'s attitude toward these constraints.\n\n- She emphasized the radical freedom and responsibility of the individual. We are "condemned to be free" because we cannot escape making choices and taking responsibility for our choices. \n\n- De Beauvoir felt that many people evade their freedom and responsibility by adopting rigid mindsets, ideologies, or conforming uncritically to social roles.\n\n- She advocated for the recognition of ambiguity in the human condition and warned against the quest for absolute rules that deny freedom and responsibility. Authentic living involves embracing ambiguity.\n\nIn summary, de Beauvoir promoted an existential ethics')}`

---

# 4. Agents and Tools

## Agents
### Construction and Management
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

## Tools
Definition: Tools are interfaces that an agent can use to interact with the world. Tools are functions that agents can use to interact with the world. These tools can be generic utilities (e.g. search), other chains, or even other agents.

Currently, tools can be loaded using the following snippet:
```python
from langchain.agents import load_tools
tool_names = []
tools = load_tools(tool_names)
```

Some tools (e.g. chains, agents) may require a base LLM to use to initialize them. 
- In that case, you can pass in an LLM as an argument:
```python
tools = load_tools(tool_names, llm=llm)
```

---