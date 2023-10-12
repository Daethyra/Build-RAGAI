# LLM Utilikit

${INTRO}
${SupportedLibraries}
${Intention : Reasoning}
${BriefResummary}

#### 1. **[OpenAI: Utilikit](./OpenAI/)**

---

A. **[Auto-Embedder](./OpenAI/Auto-Embedder)**

Provides an automated pipeline for retrieving embeddings from [OpenAIs `text-embedding-ada-002`](https://platform.openai.com/docs/guides/embeddings) and upserting them to a [Pinecone index](https://docs.pinecone.io/docs/indexes).

- **[`pinembed.py`](./OpenAI/Auto-Embedder/pinembed.py)**: A Python module to easily automate the retrieval of embeddings from OpenAI and storage in Pinecone.

---

B. **[Prompts](./OpenAI/Prompts/)**

There are three main prompt types, [multi-shot](./OpenAI/Prompts/multi-shot), [system-role](./OpenAI/Prompts/system-role), [user-role](./OpenAI/Prompts/user-role).

Please also see the [prompt-cheatsheet](./OpenAI/Prompts/prompt-cheatsheet.md).

- **[Cheatsheet](./OpenAI/Prompts/prompt-cheatsheet.md)**: @Daethyra's go-to prompts.

- **[multi-shot](./OpenAI/Prompts/multi-shot)**: Prompts, with prompts inside them. 
It's kind of like a bundle of Matryoshka prompts!

- **[system-role](./OpenAI/Prompts/system-role)**: Steer your LLM by shifting the ground it stands on.

- **[user-role](./OpenAI/Prompts/user-role)**: Markdown files for user-role prompts.

---

#### 2. **[LangChain: Pluggable Components](./LangChain/)**

---

A. **[`stateful_chatbot.py`](./LangChain/Retrieval-Agents/stateful_chatbot.py)**

This module offers a set of functionalities for conversational agents in LangChain. Specifically, it provides:

- Argument parsing for configuring the agent
- Document loading via `PyPDFDirectoryLoader`
- Text splitting using `RecursiveCharacterTextSplitter`
- Various embeddings options like `OpenAIEmbeddings`, `CacheBackedEmbeddings`, and `HuggingFaceEmbeddings`

**Potential Use Cases:** For developing conversational agents with advanced features.

---

B. **[`qa_local_docs.py`](./LangChain/Retrieval-Agents/qa_local_docs.py)**

This module focuses on querying local documents and employs the following features:

- Environment variable loading via `dotenv`
- Document loading via `PyPDFLoader`
- Text splitting through `RecursiveCharacterTextSplitter`
- Vector storage options like `Chroma`
- Embedding options via `OpenAIEmbeddings`

**Potential Use Cases:** For querying large sets of documents efficiently.

---

These modules are designed to be extensible and can be easily integrated into your LangChain projects.

---

#### 3. **[HuggingFace: Pluggable Components](./HuggingFace/)**

A. **[`integrable_captioner.py`](./HuggingFace\image_captioner\integrable_image_captioner.py)**

This module focuses on generating captions for images using Hugging Face's transformer models. Specifically, it offers:

- Model and processor initialization via the `ImageCaptioner` class
  - Image loading through the `load_image` method
  - Asynchronous caption generation using the `generate_caption` method
  - Caption caching for improved efficiency
  - Device selection (CPU or GPU) based on availability

**Potential Use Cases:** For generating accurate and context-appropriate image captions.

---

<div style="display: flex; flex-direction: row;">
  <div style="flex: 1;">
    <img src=".github\mindmap_2023-10-07.jpg" alt="Creation Date: Oct 7th, 2023" width="256"/>
  </div>
  <div style="flex: 1; display: flex; flex-direction: column;">
    <img src=".github\pie_chart.jpg" alt="Creation Date: Oct 7th, 2023" width="450"/>
    <img src=".github\bar_graph.jpg" alt="Creation Date: Oct 7th, 2023" width="450"/>
  </div>
</div>

### - [LICENSE - GNU Affero GPL](./LICENSE)