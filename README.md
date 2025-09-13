# Build-RAGAI

## Description
This project is meant to teach one how to build custom AI Agents using LangChain's Python library.

Code created since 9/12/25 will be built on LangChain's v1.0. The code in [OpenAI](./src/opai/) and [Transformers](./src/transformers/) use an old LangChain API version and are not planned to be updated to the new API. Old code snippets and notebooks will be preserved in place.

Moving forward, only DeepSeek will be supported, however, the DeepSeek API is just like the OpenAI SDK, and for the most part, you can replace "DeepSeek" with "OpenAI" and the model parameter in the code and it'll still work.

This project includes code snippets and jupyter notebooks that you can augment or outright copy.

If you're new to building AI-powered applications, I suggest you start by playing with and executing the code in the [the DeepSeek notebook](./src/langchain/notebooks/). Go through the cells and play with the code.

## Table of Contents
Below you'll find links to, and descriptions of, sections of this project for easy navigation.

This README:
- [Getting Started](#getting-started)
- [Installation](#installation)
- [License](#license)

[LangChain](./src/langchain/):
- [Code Snippets](./src/langchain/codesnippets/ "Directory"): Here you'll find pluggable Python components.
  - - ---------- OLD API ----------
  - [bufferwindow_memory.py](./src/langchain/codesnippets/bufferwindow_memory.py "Code Snippet"): A simple memory component that can be used in a LangChain conversation.
  - [chatopenai.py](./src/langchain/codesnippets/chatopenai.py "Code Snippet"): A simple LLM component that can be used to return chat messages.
  - [multi_queryvector_retrieval.py](./src/langchain/codesnippets/multi_queryvector_retrieval.py "Code Snippet"): An advanced retriever component that combines the power of multi-querying and multi-vector retrieval.

- [Notebooks](./src/langchain/notebooks/ "Directory"): Here you'll find Jupyter notebooks that guide you through the use of many different LangChain classes.
  - [DeepSeek](./src/langchain/notebooks/DeepSeek.ipynb "Notebook"): Create your first LangChain v1.0 Agent that has access to memory and tools, including weather & web searching, and a calculator.
  - ---------- OLD API ----------
  - [MergedDataLoader](./src/langchain/notebooks/rag_MergedDataLoader.ipynb "Notebook"): Learn how to embed and query multiple data sources via `MergedDataLoader`. In this notebook, we learn how to clone GitHub repositories and scrape web documentation before embedding them into a vectorstore which we then use as a retriever. By the end of it, you should be comfortable using whatever sources as context in your own RAG projects.
  - [Custom Tools](./src/langchain/notebooks/agentexecutor_custom_tools.ipynb "Notebook"): Learn how to create and use custom tools in LangChain agents.
  - [Image Generation and Captioning + Video Generation](./src/langchain/notebooks/image_generation_and_captioning.ipynb "Notebook"): Learn to create an agent that chooses which generative tool to use based on your prompt. This example begins with the agent generating an image after refining the user's prompt.
  - [LangSmith Walkthrough](./src/langchain/notebooks/langsmith_walkthrough.ipynb "Notebook"): Learn how to use LangSmith tracing and pull prompts fromt he LangSmith Hub.
  - [Retrieval Augmented Generation](./src/langchain/notebooks/rag_basics.ipynb "Notebook"): Get started with Retrieval Augmented Generation to enhance the performance of your LLM.
  - [MongoDB RAG](./src/langchain/notebooks/rag_mongoDB.ipynb "Notebook"): Perform similarity searching, metadata filtering, and question-answering with MongoDB.
  - [Pinecone and ChromaDB](./src/langchain/notebooks/rag_pinecone_chromadb.ipynb "Notebook"): A more basic but thorough walkthrough of performing retrieval augmented generation with two different vectorstores.
  - [FAISS and the HuggingFaceHub](./src/langchain/notebooks/rag_privacy_faiss_huggingfacehub.ipynb "Notebook"): Learn how to use FAISS indexes for similarity search with HuggingFaceHub embeddings. This example is a privacy friendly option, as everything runs locally. No GPU required!
  - [Runnables and Chains (LangChain Expression Language)](./src/langchain/notebooks/runnables_and_chains.ipynb "Notebook"): Learn the difference of and how to use Runnables and Chains in LangChain. Here you'll dive deep into their specifics.

- [End to End Examples](./src/langchain/packages/ "Directory"): Here you'll find scripts made to work out of the box.
  - - ---------- OLD API ----------
  - [RAG with Agents](./src/langchain/packages/rag-with-agents/ "Directory"): Learn to use Agents for RAG.
    - [Streamlit Chatbot](./src/langchain/packages/chatbots/streamlit/ "Directory"): A simple Streamlit chatbot using OpenAI.
    - [Directory Loader](./src/langchain/packages/rag-with-agents/directoryloader/README.md "Directory"): Use the `DirectoryLoader` class to load files for querying.
    - [PyPDF Directory Loader](./src/langchain/packages/rag-with-agents/pypdfdirectoryloader/README.md "Directory"): Use the `PypdfDirectoryLoader` class to load files for querying.
    - [Facebook AI Similarity Search](./src/langchain/packages/rag-with-agents/faiss_retriever.py "Directory"): Use the `FacebookAISimilaritySearch` class to load files for querying.
    - [Vectorstore RAG](./src/langchain/packages/vectorstore-rag/ "Directory"): Learn how to use vectorstores in LangChain.
    - [Pinecone](./src/langchain/packages/vectorstore-rag/pinecone/README.md "Directory"): Use a `Pinecone` vector database "Index" as a retriever and chat with your documents. 

[OpenAI](./src/opai/):
- [Code Snippets](./src/opai/codesnippets/ "Directory"): Here you'll find code snippets using the OpenAI Python library.
  - [Text to Speech](./src/opai/codesnippets/tts.py "Code Snippet"): Use the Whisper API to generate speech from text.
  - [Notebooks](./src/opai/notebooks/ "Directory"): Here you'll find Jupyter notebooks that show you how to use the OpenAI Python library.
    - [Retrieval Augmented Generation](./src/opai/notebooks/gen-qa-openai.ipynb "Notebook"): Get started with Retrieval Augmented Generation and Pinecone to enhance the performance of your LLM.

[Transformers](./src/transformers/):
- [Code Snippets](./src/transformers/codesnippets/ "Directory"): Here you'll find code snippets using the Transformers Python library.
  - [Dolphin Mixtral](./src/transformers/codesnippets/dolphin_mixtral.py "Code Snippet"): A simple function to generate text using `pipeline`.
  - [Notebooks](./src/transformers/notebooks/ "Directory"): Here you'll find Jupyter notebooks that show you how to use the Transformers Python library.
    - [Automatic Speech Recognition](./src/transformers/notebooks/asr_pipelines.ipynb "Notebook"): Transcribe speech using Whisper-v3 in a Gradio demo.
  - [Packages](./src/transformers/packages/ "Directory"): Here you'll find CLI applications.
    - [Audio Transcription](./src/transformers/packages/audiotranscription/ "Directory"): 
      - [MicTranscription](./src/transformers/packages/audiotranscription/mictranscription/ "CLI App"): Transcribe audio using a microphone.
      - [Task Creation](./src/transformers/packages/audiotranscription/taskcreation/ "CLI App"): Generates tasks based on transcribed audio.
    - [Train with Accelerate](./src/transformers/packages/trainwithaccelerate/ "Directory"): Fine tune a sequence classification model using Accelerate to make things go extra fast.

---

## Getting Started

### Installation

#### Local Code Execution and Testing
Start by navigating to the root directory of this project.

For Unix, run `pip install virtualenv && virtualenv .venv && source .venv/bin/activate`

Or for Windows, run `pip install virtualenv ; virtualenv .venv ; .venv/Scripts/activate`

This project is developed using [PDM](https://pdm.fming.dev/). It is not recommended to install dependencies this way, especially for the new notebooks built on LangChain v1.0 because PDM will install the latest *stable* release. I recommend creating virtual environments within the notebooks themselves.


If you still wish to use PDM, you can install it using `pip`:

`pip install -U pdm`

Then, install the dependencies using PDM:

`pdm install`

This command will create a virtual environment in `.venv` and install the dependencies in that environment. If you're on macOS or Linux, you can run `source .venv/bin/activate` to activate the environment. Otherwise, you can run the command `.venv/Scripts/activate` or `.venv/Scripts/activate.ps1` to activate the environment.

By using a virtual environment we avoid cross contaminating our global Python environment.

Once our virtual environment is set up we need to select it as our kernel for the Jupyter Notebook. If you're in VSCode, you can do this at the top right of the notebook. If you're using a different IDE, you'll need to look for setup help online.

When selecting the kernel, ensure you choose the one that's located inside of the `.venv` directory, and not the global Python environment.

## [MIT LICENSE](./LICENSE "MIT")
