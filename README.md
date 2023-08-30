
# OpenAI Utility Toolkit (OUT)

## Welcome to the OpenAI Utility Toolkit (OUT)

Your one-stop destination for enhancing your interaction with OpenAI models. This toolkit has grown to include a multitude of utilities that offer additional functionality and ease-of-use.

---

## Contents

1. **PinEbed.py**: A Python module to easily automate the retrieval of embeddings from OpenAI and storage in Pinecone.
2. **main.py**: The entry point for the GUI application.
3. **gui.py**: Contains the GUI logic and interface.
4. **openai_chat.py**: A Python class designed for chat-based interaction with OpenAI models.
5. **TLDR.md**: A Markdown file providing a quick overview of the project and its components.

---

## Detailed Description

### [PinEbed.py](Auto-Embedder/PinEbed.py)

This module provides a class `PineconeHandler` that handles data stream embedding and storage in Pinecone. It uses OpenAI to generate embeddings for text data and stores these embeddings in a Pinecone index.

#### How to Use

```python
from PinEbed import PineconeHandler

pinecone_handler = PineconeHandler()
# Your code to fetch data
pinecone_handler.process_data(your_data)
```

### [main.py](./main.py)

This is the entry point for the GUI application. It sets up the environment and initiates the GUI.

#### How to Use

Run this file to start the GUI application.

```bash
python main.py
```

### [gui.py](./gui.py)

This file contains the GUI logic and interface. It uses Python's `Tkinter` to create the GUI.

#### How to Use

This file is imported and used in `main.py`.

### [openai_chat.py](./openai_chat.py)

A Python class designed for chat-based interaction with OpenAI models. You can set the model and temperature for the chat.

#### How to Use

```python
from openai_chat import OpenAI_Chat

chat = OpenAI_Chat(model='gpt-4', temperature=0)
```

### [TLDR.md](./TLDR.md)

Provides a quick overview of the project and its components. Useful for getting a snapshot view of what this toolkit offers.

---
