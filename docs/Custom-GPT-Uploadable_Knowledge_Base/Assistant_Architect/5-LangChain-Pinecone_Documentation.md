# Langchain | Pinecone Vector Database Implementation 

[Read More](https://python.langchain.com/docs/integrations/vectorstores/pinecone)

### Pinecone is a vector database with broad functionality.

This notebook shows how to use functionality related to the Pinecone vector database.

To use Pinecone, you must have an API key. Here are the installation instructions.

pip install pinecone-client openai tiktoken langchain

import getpass
import os

os.environ["PINECONE\_API\_KEY"] = getpass.getpass("Pinecone API Key:")

os.environ["PINECONE\_ENV"] = getpass.getpass("Pinecone Environment:")

We want to use OpenAIEmbeddings so we have to get the OpenAI API Key.

os.environ["OPENAI\_API\_KEY"] = getpass.getpass("OpenAI API Key:")

from langchain.document\_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text\_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone

from langchain.document\_loaders import TextLoader

loader = TextLoader("../../modules/state\_of\_the\_union.txt")
documents = loader.load()
text\_splitter = CharacterTextSplitter(chunk\_size=1000, chunk\_overlap=0)
docs = text\_splitter.split\_documents(documents)

embeddings = OpenAIEmbeddings()

import pinecone

# initialize pinecone
pinecone.init(
 api\_key=os.getenv("PINECONE\_API\_KEY"), # find at app.pinecone.io
)

index\_name = "langchain-demo"

# First, check if our index already exists. If it doesn't, we create it
if index\_name not in pinecone.list\_indexes():
 # we create a new index
 pinecone.create\_index(name=index\_name, metric="cosine", dimension=1536)
# The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
docsearch = Pinecone.from\_documents(docs, embeddings, index\_name=index\_name)

# if you already have an index, you can load it like this
# docsearch = Pinecone.from\_existing\_index(index\_name, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity\_search(query)

print(docs[0].page\_content)

Adding More Text to an Existing Index​

More text can embedded and upserted to an existing Pinecone index using the add\_texts function

index = pinecone.Index("langchain-demo")
vectorstore = Pinecone(index, embeddings.embed\_query, "text")

vectorstore.add\_texts("More text!")

Maximal Marginal Relevance Searches​

In addition to using similarity search in the retriever object, you can also use mmr as retriever.

retriever = docsearch.as\_retriever(search\_type="mmr")
matched\_docs = retriever.get\_relevant\_documents(query)
for i, d in enumerate(matched\_docs):
 print(f"\n## Document {i}\n")
 print(d.page\_content)

Or use max\_marginal\_relevance\_search directly:

found\_docs = docsearch.max\_marginal\_relevance\_search(query, k=2, fetch\_k=10)
for i, doc in enumerate(found\_docs):
 print(f"{i + 1}.", doc.page\_content, "\n")
