### Todo list

[README]

- Add intro
  - Clearly define: [Utilikit, Pluggable/Components, multi-shot, zero-shot,]
    - create summarization of prompt reusability, and component extendability
  - Then, clearly state the intention of the repository. : Provide Reasoning, I want this to be a nexus of information to empower my LLMs moving forward. By continually updating this repository as a codebase and conglomeration of documentation, it may serve as a `git clone`able neuron for machine learning models.
  - Finally, provide one to two brief statements to close out and resummarize

---

[GitHub]

- Clean all of my Jupyter notebook Gists to create an agent
  - With production-grade code stored locally, it may be useful as a codebase for LLM agents.
  - By supplying enough production-grade code that can be repurposed for our intended use case, we can steer our Generator towards desirable outcomes.
  - [Objective] : "Build a knowledgebase for leverage in programming production-grade code, specifically related to the creation of more machine learning code, much like it."
  - [Thoughts] : "By creating a codebase of repurposable Python modules,
    we can use LangChain to query the top_k results for functions to serve
    contextual needs."

---

[LangChain]

- ~~langchain_conv_agent.py~~

  - ~~Lacks single execution runnability~~
  - ~~Fix by removing argparsing and implement default settings, with a configuration file~~
  - Config file settings:
    - Embedding Engine: [OpenAI, HuggingFace, etc.]
  - ***Lacks .env var loading(API keys, model names[OpenAI, HuggingFace])***
  - Ambiguity regarding (EmbeddingManager and DocumentRetriever)
    - Needs comments and to load via .env file
      - Differentiate EmbeddingManager and DocumentRetriever by explaining how they're implemented into the pipeline stream created by the module.
      - One generates embeddings
      - `DocumentRetriever` queries them locally
        (HF model is cached after first download. Therefore, all runs after the first,
        are entirely local since we're using ChromaDB)
- qa_local_docs.py

  - ~~Doesn't automatically collect and generate embeddings for the data folder~~
  - ~~To ensure automation, create a first-run / boot-up process~~

  1. Move the `PDFProcessor` class to a separate file to increase modularity and maintainability.
  2. Use dependency injection to pass in the necessary objects to the `PDFProcessor` class instead of initializing them in the constructor. This will increase modularity and make the class more testable.
  3. Use a logger instead of `print` statements to log errors and other messages. This will make the code more maintainable and scalable.
  4. Use constants or configuration files to store environment variables and other configuration settings. This will make the code more maintainable and scalable.
  5. Use type hints and docstrings to improve readability and maintainability of the code.
  6. Refactor the `perform_similarity_search` method to use a more efficient algorithm for similarity search, such as Locality-Sensitive Hashing (LSH) or Approximate Nearest Neighbors (ANN). This will increase scalability and reliance of the code.
  7. Refactor the `load_pdfs_from_directory` method to use a more efficient PDF parsing library, such as PyPDF2 or pdfminer. This will increase scalability and reliance of the code.
  8. Refactor the `answer_question` method to use a more advanced question answering model, such as BERT or T5. This will increase the accuracy and reliability of the answers.
  9. Use version control to track changes to the code and collaborate with other developers. This will increase maintainability and reliance of the code.
  10. Write unit tests to ensure that the code works as expected and to catch regressions. This will increase maintainability and reliance of the code.

---

[OpenAI]

- ~~Auto-Embedder~~
  - ~~Requires testing~~
    - ~~test.py requires updates~~
- ~~[Task]:Update test.py and run~~

---
