# Build-RAGAI

## Description
This project seeks to teach you how to build Python applications with generative AI functionality by using the LangChain and Transformers libraries.

While there is a section for [OpenAI](./src/opai/), most of the code that previously existed there has been repurposed and integrated with either the [LangChain](./src/langchain/) or [Transformers](./src/transformers/) libraries. This project includes code snippets, packages examples, and jupyter notebooks that you can augment, copy, or learn from respectively.

If you're new to building AI-powered applications, I suggest you start by playing with and executing the code in the [LangChain notebooks](./src/langchain/notebooks/). Seeing the code in action, editing it yourself, and creatively brainstorming new ideas is the best way to learn.

## Table of Contents
Below you'll find links to, and descriptions of, sections of this project for easy navigation.

This README:
- [Getting Started](#getting-started)
- [Installation](#installation)
- [License](#license)

[LangChain](./src/langchain/):
- [Code Snippets](./src/langchain/codesnippets/ "Directory"): Here you'll find pluggable Python components.
  - [bufferwindow_memory.py](./src/langchain/codesnippets/bufferwindow_memory.py "Code Snippet"): A simple memory component that can be used in a LangChain conversation.
  - [chatopenai.py](./src/langchain/codesnippets/chatopenai.py "Code Snippet"): A simple LLM component that can be used to return chat messages.
  - [multi_queryvector_retrieval.py](./src/langchain/codesnippets/multi_queryvector_retrieval.py "Code Snippet"): An advanced retriever component that combines the power of multi-querying and multi-vector retrieval.

- [Notebooks](./src/langchain/notebooks/ "Directory"): Here you'll find Jupyter notebooks that guide you through the use of many different LangChain classes.
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
This project is developed using [PDM](https://pdm.fming.dev/). You can install PDM using `pip`:

Start by navigating to the root directory of this project, then run:

```bash
pip install -U pdm
```

Then you'll need to install the dependencies using PDM:

```bash
pdm install
```

This command will create a virtual environment in `.venv` and install the dependencies in that environment. If you're on macOS or Linux, you can run `source .venv/bin/activate` to activate the environment. Otherwise, you can run the command `.venv/Scripts/activate` or `.venv/Scripts/activate.ps1` to activate the environment.

By using a virtual environment we avoid cross contaminating our global Python environment.

Once our virtual environment is set up we need to select it as our kernel for the Jupyter Notebook. If you're in VSCode, you can do this at the top right of the notebook. If you're using a different IDE, you'll need to look for setup help online.

When selecting the kernel, ensure you choose the one that's located inside of the `.venv` directory, and not the global Python environment.

---

### Test Your First Notebook
If you're totally new to building AI powered applications with access to external data, specifically retrieval augmented generation, check out the [RAG Basics](./src/langchain/notebooks/rag_basics.ipynb "Starter RAG Notebook for learning") notebook. It's the most straightforward notebook, and its concepts are built upon in every other 'RAG' notebook.

#### Google Colab

Click the badge below to open the RAG Basics notebook in Colab.

<a target="_blank" href="https://colab.research.google.com/github/Daethyra/Build-RAGAI/blob/master/src/langchain/notebooks/rag_basics.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 'rag_basics.ipynb' In Colab"/>
</a>

---

## [LICENSE](./LICENSE "GNU Affero GPL")
