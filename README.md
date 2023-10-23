# LLM Utilikit

🤍Welcome to the Utilikit, a library of Python modules designed to supercharge your large-language-model projects. Whether you're just getting started or looking to enhance an existing project, this library offers a rich set of pluggable components and a treasure trove of large language model prompts and templates. And I invite all proompters to enrich this toolkit with their own prompts, templates, and Python modules.

## Supported libraries:
- OpenAI
- LangChain
- HuggingFace
- Pinecone

This project aims to solve two key challenges faced by developers and data scientists alike: the need for a quick start and the desire for modular, reusable components. This library addresses these challenges head-on by offering a curated set of Python modules that can either serve as a robust starting point for new projects or as plug-and-play components to elevate existing ones.

## 0. **[Prompts](./Prompts/)**

There are three main prompt types, [multi-shot](./Prompts/multi-shot), [system-role](./Prompts/system-role), [user-role](./Prompts/user-role).

Please also see the [prompt-cheatsheet](./Prompts/prompt-cheatsheet.md).

- **[Cheatsheet](./Prompts/prompt-cheatsheet.md)**: @Daethyra's go-to prompts.

- **[multi-shot](./Prompts/multi-shot)**: Prompts, with prompts inside them. 
It's kind of like a bundle of Matryoshka prompts!

- **[system-role](./Prompts/system-role)**: Steer your LLM by shifting the ground it stands on.

- **[user-role](./Prompts/user-role)**: Markdown files for user-role prompts.

## 1. **[OpenAI](./OpenAI/)**

A. **[Auto-Embedder](./OpenAI/Auto-Embedder)**

Provides an automated pipeline for retrieving embeddings from [OpenAIs `text-embedding-ada-002`](https://platform.openai.com/docs/guides/embeddings) and upserting them to a [Pinecone index](https://docs.pinecone.io/docs/indexes).

- **[`pinembed.py`](./OpenAI/Auto-Embedder/pinembed.py)**: A Python module to easily automate the retrieval of embeddings from OpenAI and storage in Pinecone.

## 2. **[LangChain](./LangChain/)**

A. **[`stateful_chatbot.py`](./LangChain/Retrieval-Augmented-Generation/qa_local_docs.py)**

This module offers a set of functionalities for conversational agents in LangChain. Specifically, it provides:

- Argument parsing for configuring the agent
- Document loading via `PDFProcessor`
- Text splitting using `RecursiveCharacterTextSplitter`
- Various embeddings options like `OpenAIEmbeddings`, `CacheBackedEmbeddings`, and `HuggingFaceEmbeddings`

**Potential Use Cases:** For developing conversational agents with advanced features.

B. **[`qa_local_docs.py`](./LangChain/Retrieval-Agents/qa_local_docs.py)**

This module focuses on querying local documents and employs the following features:

- Environment variable loading via `dotenv`
- Document loading via `PyPDFLoader`
- Text splitting through `RecursiveCharacterTextSplitter`
- Vector storage options like `Chroma`
- Embedding options via `OpenAIEmbeddings`

**Potential Use Cases:** For querying large sets of documents efficiently.

### 3. **[HuggingFace](./HuggingFace/)**

A. **[`integrable_captioner.py`](./HuggingFace\image_captioner\integrable_image_captioner.py)**

This module focuses on generating captions for images using Hugging Face's transformer models. Specifically, it offers:

- Model and processor initialization via the `ImageCaptioner` class
  - Image loading through the `load_image` method
  - Asynchronous caption generation using the `generate_caption` method
  - Caption caching for improved efficiency
  - Device selection (CPU or GPU) based on availability

**Potential Use Cases:** For generating accurate and context-appropriate image captions.

## Installation

Distribution as a package for easy installation and integration is planned, however that *not* currently in progress.

---

<div style="display: flex; flex-direction: row;">
  <div style="flex: 1;">
    <img src=".github\2023-10-18_Mindmap.jpg" alt="Creation Date: Oct 7th, 2023" width="768"/>
  </div>
</div>

### - [LICENSE - GNU Affero GPL](./LICENSE)