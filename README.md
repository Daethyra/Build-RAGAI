# OpenAI Utility Toolkit (OUT)
## Welcome to the OpenAI Utility Toolkit (OUT) 
Your one-stop destination for enhancing your interaction with OpenAI models. Originally, this repository housed a single module, but I've now expanded it to include a multitude of utilities that offer additional functionality and ease-of-use.

# Contents
BlindProgamming
The BlindProgamming module is designed to help users solve their code's problems with the assistance of OpenAI's GPT models. 

For each user message, the module internally creates three separate solutions to solve the user's problem, then merges all of the best aspects of each solution into a master solution, that has its own set of enhancements and supplementary functionality. This meticulous, accuracy ensuring AI programming assistant works step-by-step to ensure the right solution is reached.

This is called 'tree-of-thoughts' prompting.

# OpenAI-Pinecone
## The OpenAI-Pinecone module is a Python script that integrates the OpenAI models with Pinecone's vector database. 

This utility initializes environment variables for Pinecone and OpenAI, validates these variables, and then sets up an argument parser. 

It also includes a retry decorator for function calls that might fail, and a function to normalize embeddings. The script ultimately creates an instance of OpenAiPinecone which can get embeddings from OpenAI's models and upsert these embeddings into Pinecone.

# Future Plans

As the author of this toolkit, I'm thrilled with the progress I've made so far, but there's so much more to come! My intention is to continue adding more back-end modules that interact with OpenAI's models and provide additional functionality for developers. 

I'm *always* open to suggestions and contributions, so don't hesitate to reach out if you've got an idea or a tool that could be a valuable addition to the OpenAI Utility Toolkit.
