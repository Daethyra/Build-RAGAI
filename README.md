# OpenAI Utility Toolkit (OUT)

## Contents
- [LICENSE - GNU Affero GPL](./LICENSE)

---

#### 1. **[Auto-Embedder](./Auto-Embedder)**

Provides an automated pipeline for retrieving embeddings from [OpenAI's `text-embedding-ada-002`](https://platform.openai.com/docs/guides/embeddings) and upserting them to a [Pinecone index](https://docs.pinecone.io/docs/indexes).

   - **[pinembed.py](./Auto-Embedder/pinembed.py)**: A Python module to easily automate the retrieval of embeddings from OpenAI and storage in Pinecone.
   - **[.env.template](./Auto-Embedder/.env.template)**: Template for environment variables.

---

#### 2. **[GPT-Prompt-Examples](./GPT-Prompt-Examples)**

There are three main prompt types, [multi-shot](GPT-Prompt-Examples/multi-shot), [system-role](GPT-Prompt-Examples/system-role), [user-role](GPT-Prompt-Examples/user-role).

Please also see the [OUT-prompt-cheatsheet](GPT-Prompt-Examples/OUT-prompt-cheatsheet.md).

   - **[Cheatsheet for quick power-prompts](./GPT-Prompt-Examples/OUT-prompt-cheatsheet.md)**: A cheatsheet for GPT prompts.
   - **[multi-shot](./GPT-Prompt-Examples/multi-shot)**: Various markdown and text files for multi-shot prompts.
   - **[system-role](./GPT-Prompt-Examples/system-role)**: Various markdown files for system-role prompts.
   - **[user-role](./GPT-Prompt-Examples/user-role)**: Markdown files for user-role prompts.
   - **[Reference Chatlogs with GPT4](./GPT-Prompt-Examples/ChatGPT_reference_chatlogs)**: Contains chat logs and shorthand prompts.
  
---

#### 3. **[LangChain: Pluggable Components](./LangChain/)**

This section describes the pre-built Python modules in the `LangChain` directory that can be side-loaded for various functionalities.

`langchain_conv_agent.py`

This module offers a set of functionalities for conversational agents in LangChain. Specifically, it provides:

- Argument parsing for configuring the agent
- Document loading via `PyPDFDirectoryLoader`
- Text splitting using `RecursiveCharacterTextSplitter`
- Various embeddings options like `OpenAIEmbeddings`, `CacheBackedEmbeddings`, and `HuggingFaceEmbeddings`

**Usage:**
To use this module, simply import the functionalities you need and configure them accordingly.

---

`query_local_docs.py`

This module focuses on querying local documents and employs the following features:

- Environment variable loading via `dotenv`
- Document loading via `PyPDFLoader`
- Text splitting through `RecursiveCharacterTextSplitter`
- Vector storage options like `Chroma`
- Embedding options via `OpenAIEmbeddings`

**Usage:**
Similar to `langchain_conv_agent.py`, you can import the functionalities you require.

These modules are designed to be extensible and can be easily integrated into your LangChain projects.

### Mindmap

<div align="center">
  <img src=".github\mindmap_10-7-23_.png" alt="Creation Date: Oct 7th, 2023" width="500"/>
</div>

---