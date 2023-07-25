# OpenAI Utility Toolkit (OUT)

## Welcome to the OpenAI Utility Toolkit (OUT)

Your one-stop destination for enhancing your interaction with OpenAI models. Originally, this repository housed a single module, but I've now expanded it to include a multitude of utilities that offer additional functionality and ease-of-use.

# Contents

## [Blind Progamming](https://github.com/Daethyra/OpenAI-Utility-Toolkit/tree/master/Blind%20Programming)

The BlindProgamming folder contains text files designed to inspire/augment users' solve their code's problems with the assistance of OpenAI's GPT models.
Includes prompt examples at the [user](https://github.com/Daethyra/OpenAI-Utility-Toolkit/blob/Daethyra-patch-1/Blind%20Programming/User-Role_Prompts.md) and [system](https://github.com/Daethyra/OpenAI-Utility-Toolkit/blob/Daethyra-patch-1/Blind%20Programming/System-Role_Prompts.md) levels, and there's currently a single [multi-shot prompt example](https://github.com/Daethyra/OpenAI-Utility-Toolkit/blob/Daethyra-patch-1/Blind%20Programming/multi-shot-prompt-example.md).

## [Auto-Embedder](https://github.com/Daethyra/OpenAI-Utility-Toolkit/blob/master/Auto-Embedder/autoembeds.py)

The OpenAI-Pinecone module is a Python script that integrates the OpenAI models with Pinecone's vector database.
This utility requires integration with other modules to function, as it is a back-end processor that requests embeddings for the user's input from OpenAI and then sends them over a Pinecone index.

## [GPT-Chatbot](https://github.com/Daethyra/OpenAI-Utility-Toolkit/blob/master/GPT-Chatbot/gui.py)

This is a standalone module that creates a GUI interface which provides functionality to send 'system' messages, reset the conversation, and automatically store the last 10 responses from the GPT model
