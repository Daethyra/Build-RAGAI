# Streamline calls to OpenAI and Pinecone | Automate the OP stack

## What's this?

This single pluggable module named [pinembed.py](pinembed.py) provides a data-pipe using the OP stack.
It automates the retrieval of vector embeddings from OpenAI's `text-embeddings-ada-002` model as well the uploading of said data to a Pinecone index.

It does the following:

- Ingests data
- Sends data to 'Ada-002' at OpenAI to receive embeddings
- Automatically [upserts](https://docs.pinecone.io/reference/upsert "Upsert documentation") received embedding data in real time

## Why should I care?

- Skip the programming!
- Provides a modular multi-class structure for isolating and using specific functionality, like asynchronous embedding retrieval.
- Eases the process of building Large Language Models
- Enables semantic similarity searches
- [Empowers](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings#:~:text=To%20see%20embeddings%20in%20action%2C%20check%20out%20our%20code%20samples "Reference Documentation"):
  - Classification
  - Topic clustering
  - Search
  - Recommendations

### Requirements

- OpenAI
- Pinecone
- Python-dotenv

## Roadmap

1) Create pseudocode for more functionality, namely further querying the Pinecone index
2) Draft Python logic for ['similarity'](https://docs.pinecone.io/reference/query) queries
3) Remove 0.3 data-stream cooldown. | This is literally an async pluggable module -- don't need that.
4) Create LangChain class on top of `DataStreamHandler` with the goal of testing it as a Question/Answering service
   * LangChain `DirectoryLoader`
5) Extend package to enable [Agents](https://www.pinecone.io/learn/series/langchain/langchain-agents/ "Agent Documentation") & [Memory](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/ "Memory Documentation") for large language models

#### Self-asked Dev-questions

- How will someone easily query their index?
  - Automating 'similarity' queries is a good starting point
- How can this module be even easier to side-load for *any* project?
- Did I properly write docstrings that accurately reflect the expected data types for Pinecone specifically? I know I checked for Ada-002.
- Is it worth having multiple data streams for different processes an end-user might have? Especially if they're an organization, with multiple keys running?
  - I'd also therefore need to make room for more keys, etc. I will use organizational ID management to help further differentiate where necessary.

## Official Reference Documentation

- [OpenAI Documentation](https://platform.openai.com/docs/guides/embeddings)
- [Embeddings API Reference](https://platform.openai.com/docs/api-reference)
- [Pinecone Example Projects](https://docs.pinecone.io/page/examples)
- [Pinecone API Reference](https://docs.pinecone.io/reference)
- [LangChain / Pinecone "Getting Startetd"](https://www.pinecone.io/learn/series/langchain/langchain-intro/)
- [LangChain Agents](https://www.pinecone.io/learn/series/langchain/langchain-agents/)
- [LangChain Conversational Memory](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)

## [LICENSE](../LICENSE)
