# Embedding Automation with OpenAI and Pinecone

This module provides an easy way to automate the retrieval of embeddings from OpenAI's `text-embedding-ada-002` model and store them in a Pinecone index. The module does the following:

- Ingests data
- Sends data to 'Ada-002' at OpenAI to receive embeddings
- Automatically upserts received embedding data in real time

## Requirements

- OpenAI
- Pinecone
- Python-dotenv
- LangChain

## Usage

1. Set up environment variables in a `.env` file.
2. Place files to be processed in the `data` directory.
3. Run `python pinembed.py`.

## Roadmap

1. ~~Create pseudocode for more functionality, namely further querying the Pinecone index.~~ (***Outside scope)***
2. ~~Draft Python logic for [&#39;similarity&#39;](https://docs.pinecone.io/reference/query) queries.~~ (***Outside scope)***
3. ~~Remove 0.3 data-stream cooldown. This is an async pluggable module -- it doesn't need that.~~ (***Outside scope)***
4. ~~Create LangChain class on top of `DataStreamHandler` with the goal of testing it as a Question/Answering service.~~ (***Outside scope)***
   * ~~LangChain `DirectoryLoader`~~
5. ~~Extend package to enable [Agents](https://www.pinecone.io/learn/series/langchain/langchain-agents/ "Agent Documentation") & [Memory](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/ "Memory Documentation") for large language models.~~

## Official Reference Documentation

- [OpenAI Documentation](https://platform.openai.com/docs/guides/embeddings)
- [Embeddings API Reference](https://platform.openai.com/docs/api-reference)
- [Pinecone Example Projects](https://docs.pinecone.io/page/examples)
- [Pinecone API Reference](https://docs.pinecone.io/reference)
- [LangChain / Pinecone Getting Started](https://www.pinecone.io/learn/series/langchain/langchain-intro/)
- [LangChain Agents](https://www.pinecone.io/learn/series/langchain/langchain-agents/)
- [LangChain Conversational Memory](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)

## [LICENSE](../LICENSE)
