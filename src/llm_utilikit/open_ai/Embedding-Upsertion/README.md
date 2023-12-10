# Embedding Automation with OpenAI and Pinecone

## OpenAI Embedding Upsertion to Pinecone without LangChain as the module liason.
- `pinembed.py` is designed by using the `openai` and `pinecone-client` Python libraries directly, rather than loading them through LangChain. This will be helpful for those familiar with the former, and not the latter.

## What is it?
This module provides an easy way to automate the retrieval of embeddings from OpenAI's `text-embedding-ada-002` model and store them in a Pinecone index. The module does the following:

- Ingests data
- Sends data to 'Ada-002' at OpenAI to receive embeddings
- Automatically upserts received embedding data in real time

### Requirements

- OpenAI
- Pinecone
- Python-dotenv
- LangChain

## Usage

1. Set up environment variables in a `.env` file.
2. Place files to be processed in the `data` directory.
3. Run `python pinembed.py`.

## Official Reference Documentation

- [OpenAI Documentation](https://platform.openai.com/docs/guides/embeddings)
- [Embeddings API Reference](https://platform.openai.com/docs/api-reference)
- [Pinecone Example Projects](https://docs.pinecone.io/page/examples)
- [Pinecone API Reference](https://docs.pinecone.io/reference)
- [LangChain / Pinecone Getting Started](https://www.pinecone.io/learn/series/langchain/langchain-intro/)
- [LangChain Agents](https://www.pinecone.io/learn/series/langchain/langchain-agents/)
- [LangChain Conversational Memory](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)