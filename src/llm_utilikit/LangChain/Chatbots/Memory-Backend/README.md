# Integratable ChromaMemory to store chat history and retrieve answers to questions from the conversation history.
A simple way to add memory to an AI-powered Python application.

## Usage Guide

### 1. Import the ChromaMemory class from the chroma_memory module:

`from chroma_memory import ChromaMemory`

### 2. Create an instance of the ChromaMemory class, passing in the required parameters:

```
model_name = "text-embedding-ada-002"
cache_dir = "/opt/llm/vectorstore/chroma"
vectorstore = Chroma("/opt/llm/vectorstore/chroma")
chroma_memory = ChromaMemory(model_name, cache_dir, max_history_len, vectorstore)
max_history_len = 100
```

The model_name parameter specifies the name of the LLM model to use, the cache_dir parameter specifies the path to the directory to cache embeddings, the max_history_len parameter specifies the maximum length of the conversation history to remember, and the vectorstore parameter specifies the vectorstore to use for similarity matching.

### 3. To store a new chat message in the conversation history, call the add_message method of the ConversationBufferWindowMemory object:

```
message = "Hello, how are you?"
chroma_memory.memory.add_message(message)
```

### 4. This will add the message to the conversation history.

To retrieve an answer to a question from the conversation history, call the retrieve method of the RetrievalQA object:

```
question = "What's your favorite color?"
answer = chroma_memory.retrieval.retrieve(question)
print(answer)
```

This will retrieve the answer to the most similar question in the conversation history to the input question.