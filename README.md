# OpenAI Utility Toolkit (OUT)

## Welcome to the OpenAI Utility Toolkit (OUT)

Your one-stop destination for enhancing your interaction with OpenAI models. This toolkit has grown to include a multitude of utilities that offer additional functionality and ease-of-use.

---

## Contents

1. **[Auto-Embedder](./Auto-Embedder)**
    - **[PinEbed.py](./Auto-Embedder/PinEbed.py)**: A Python module to easily automate the retrieval of embeddings from OpenAI and storage in Pinecone.
    - **[.env.template](./Auto-Embedder/.env.template)**: Template for environment variables.

2. **[Basic-GPT-GUI](./Basic-GPT-GUI)**
    - **[main.py](./Basic-GPT-GUI/main.py)**: The entry point for the GUI application.
    - **[gui.py](./Basic-GPT-GUI/src/gui.py)**: Contains the GUI logic and interface using Python's `Tkinter`.
    - **[openai_chat.py](./Basic-GPT-GUI/src/openai_chat.py)**: A Python class designed for chat-based interaction with OpenAI models.
    - **[.env.template](./Basic-GPT-GUI/.env.template)**: Template for environment variables.
    - **[requirements.txt](./Basic-GPT-GUI/requirements.txt)**: List of required Python packages.

3. **[GPT-Prompt-Examples](./GPT-Prompt-Examples)**
    - **[OUT-prompt-cheatsheet.md](./GPT-Prompt-Examples/OUT-prompt-cheatsheet.md)**: A cheatsheet for GPT prompts.
    - **[TLDR.md](./GPT-Prompt-Examples/TLDR.md)**: A Markdown file providing a quick overview of the project and its components.
    - **[ChatGPT_reference_chatlogs](./GPT-Prompt-Examples/ChatGPT_reference_chatlogs)**: Contains chat logs and shorthand prompts.
    - **[multi-shot](./GPT-Prompt-Examples/multi-shot)**: Various markdown and text files for multi-shot prompts.
    - **[system-role](./GPT-Prompt-Examples/system-role)**: Various markdown files for system-role prompts.
    - **[user-role](./GPT-Prompt-Examples/user-role)**: Markdown files for user-role prompts.

<div align="center">
  <img src=".github/mindmap.png" alt="Mindmap from 8-30-23" width="500"/>
</div>

---

## Detailed Description

### [Auto-Embedder](./Auto-Embedder)

#### [PinEbed.py](./Auto-Embedder/PinEbed.py)

This module provides a class `PineconeHandler` that handles data stream embedding and storage in Pinecone. It uses OpenAI to generate embeddings for text data and stores these embeddings in a Pinecone index.

### [Basic-GPT-GUI](./Basic-GPT-GUI)

#### [main.py](./Basic-GPT-GUI/main.py)

This is the entry point for the GUI application. It sets up the environment and initiates the GUI.

#### [gui.py](./Basic-GPT-GUI/src/gui.py)

This file contains the GUI logic and interface. It uses Python's `Tkinter` to create the GUI.

#### [openai_chat.py](./Basic-GPT-GUI/src/openai_chat.py)

A Python class designed for chat-based interaction with OpenAI models. You can set the model and temperature for the chat.

### [GPT-Prompt-Examples](./GPT-Prompt-Examples)

Provides a collection of prompt examples and guidelines for GPT. Includes cheatsheets, chat logs, and various prompt types.

---
