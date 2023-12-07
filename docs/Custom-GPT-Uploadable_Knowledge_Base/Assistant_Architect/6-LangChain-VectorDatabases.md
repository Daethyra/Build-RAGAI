# Vector Stores: ChromaDB
Install Chroma with:
`pip install chromadb`

Chroma runs in various modes. See below for examples of each integrated with LangChain. 
- `in-memory` : in a python script or jupyter notebook 
- `in-memory with persistance` : in a script or notebook and save/load to disk 
- `in a docker container` : as a server running your local machine or in the cloud

Like any other database, you can: 
- `.add`, `.get`, `.update`, `.upsert`, `.delete`, `.peek`, and `.query` runs the similarity search.

View full docs at [docs](https://docs.trychroma.com/reference/Collection). To access these methods directly, you can do `._collection.method()`

## Basic Example
In this basic example, we take the most recent State of the Union Address, split it into chunks, embed it using an open-source embedding model, load it into Chroma, and then query it.
```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# load the document and split it into chunks
loader = TextLoader("../../modules/state_of_the_union.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

# query it
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)
```

## Basic Example Extended: Add Saving to Disk
Extending the previous example, if you want to save to disk, simply initialize the Chroma client and pass the directory where you want the data to be saved to.

`Caution`: Chroma makes a best-effort to automatically save data to disk, however multiple in-memory clients can stomp each other’s work. As a best practice, only have one client per path running at any given time.
```python
# save to disk
db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
docs = db2.similarity_search(query)

# load from disk
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db3.similarity_search(query)
print(docs[0].page_content)
```

## Passing a Chroma Client into LangChain
You can also create a Chroma Client and pass it to LangChain. This is particularly useful if you want easier access to the underlying database.

You can also specify the collection name that you want LangChain to use.

```python
import chromadb

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("collection_name")
collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="collection_name",
    embedding_function=embedding_function,
)

print("There are", langchain_chroma._collection.count(), "in the collection")
```

## Chroma using a Docker Container
You can also run the Chroma Server in a Docker container separately, create a Client to connect to it, and then pass that to LangChain.

Chroma has the ability to handle multiple `Collections` of documents, but the LangChain interface expects one, so we need to specify the collection name. The default collection name used by LangChain is “langchain”.

Here is how to clone, build, and run the Docker Image: `git clone git@github.com:chroma-core/chroma.git`
Edit the `docker-compose.yml` file and add `ALLOW_RESET=TRUE` under `environment`
```
    ...
    command: uvicorn chromadb.app:app --reload --workers 1 --host 0.0.0.0 --port 8000 --log-config log_config.yml
    environment:
      - IS_PERSISTENT=TRUE
      - ALLOW_RESET=TRUE
    ports:
      - 8000:8000
    ...
```
Then run `docker-compose up -d --build`
```
# create the chroma client
import uuid

import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(settings=Settings(allow_reset=True))
client.reset()  # resets the database
collection = client.create_collection("my_collection")
for doc in docs:
    collection.add(
        ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
    )

# tell LangChain to use our client and collection name
db4 = Chroma(
    client=client,
    collection_name="my_collection",
    embedding_function=embedding_function,
)
query = "What did the president say about Ketanji Brown Jackson"
docs = db4.similarity_search(query)
print(docs[0].page_content)
```

## Update and Delete
While building toward a real application, you want to go beyond adding data, and also update and delete data.

Chroma has users provide `ids` to simplify the bookkeeping here. `ids` can be the name of the file, or a combined has like `filename_paragraphNumber`, etc.

Chroma supports all these operations - though some of them are still being integrated all the way through the LangChain interface. Additional workflow improvements will be added soon.

Here is a basic example showing how to do various operations:
```
# create simple ids
ids = [str(i) for i in range(1, len(docs) + 1)]

# add data
example_db = Chroma.from_documents(docs, embedding_function, ids=ids)
docs = example_db.similarity_search(query)
print(docs[0].metadata)

# update the metadata for a document
docs[0].metadata = {
    "source": "../../modules/state_of_the_union.txt",
    "new_value": "hello world",
}
example_db.update_document(ids[0], docs[0])
print(example_db._collection.get(ids=[ids[0]]))

# delete the last document
print("count before", example_db._collection.count())
example_db._collection.delete(ids=[ids[-1]])
print("count after", example_db._collection.count())
```

## Integrate OpenAI Embeddings
Many people like to use OpenAIEmbeddings, here is how to set that up.
```
from getpass import getpass

from langchain.embeddings.openai import OpenAIEmbeddings

OPENAI_API_KEY = getpass()

import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = OpenAIEmbeddings()
new_client = chromadb.EphemeralClient()
openai_lc_client = Chroma.from_documents(
    docs, embeddings, client=new_client, collection_name="openai_collection"
)

query = "What did the president say about Ketanji Brown Jackson"
docs = openai_lc_client.similarity_search(query)
print(docs[0].page_content)
```

## Perform Similarity Search with Scores
The returned distance score is cosine distance. Therefore, a lower score is better.
`docs = db.similarity_search_with_score(query)`
`docs[0]`

---

# Vector Stores: Pinecone with Langchain

To use Pinecone, you must have an API key. Here are the installation instructions.

`pip install pinecone-client openai tiktoken langchain`

```
import getpass
import os

os.environ["PINECONE_API_KEY"] = getpass.getpass("Pinecone API Key:")

os.environ["PINECONE_ENV"] = getpass.getpass("Pinecone Environment:")
```

We want to use OpenAIEmbeddings so we have to get the OpenAI API Key.

```
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone

from langchain.document_loaders import TextLoader

loader = TextLoader("../../modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
```

```
import pinecone

# initialize pinecone
pinecone.init(
 api_key=os.getenv("PINECONE_API_KEY"), # find at app.pinecone.io
)

index_name = "langchain-demo"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
 # we create a new index
 pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
# The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)

print(docs[0].page_content)
```

## Adding More Text to an Existing Index​

More text can be embedded and upserted to an existing Pinecone index using the `add_texts` function:

```
index = pinecone.Index("langchain-demo")
vectorstore = Pinecone(index, embeddings.embed_query, "text")

vectorstore.add_texts("More text!")
```

## Maximal Marginal Relevance Searches​

In addition to using similarity search in the retriever object, you can also use mmr as retriever.

```
retriever = docsearch.as_retriever(search_type="mmr")
matched_docs = retriever.get_relevant_documents(query)
for i, d in enumerate(matched_docs):
 print(f"\n## Document {i}\n")
 print(d.page_content)
```

Or use max_marginal_relevance_search directly:

```
found_docs = docsearch.max_marginal_relevance_search(query, k=2, fetch_k=10)
for i, doc in enumerate(found_docs):
 print(f"{i + 1}.", doc.page_content, "\n")
```