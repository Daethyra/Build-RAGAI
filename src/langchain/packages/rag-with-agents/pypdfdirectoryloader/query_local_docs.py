import os
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import RunnableParallel
from langchain.utils.text_splitter import RecursiveCharacterTextSplitter
from langchain.hub import Hub

# Set API Key
os.environ["OPENAI_API_KEY"] = ""
# Initialize the hub
hub = Hub()

try:
    # Load PDF documents using PyPDFLoader with text splitting
    pdf_loader = PyPDFDirectoryLoader(
        "docs",
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=2048, chunk_overlap=256
        ),
    )
    # load documents and split into chunks
    pdf_documents = pdf_loader.load_and_split()

    # Initialize OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Initialize Chroma vector store and embed the PDF documents
    vector_store = Chroma.from_documents(pdf_documents, embeddings)

    # Initialize ChatOpenAI with gpt-3.5-turbo-1106 model and temperature of 0.25
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.25)

    # Function to format the documents
    def format_documents(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    formatted_docs = format_documents(pdf_documents)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        top_k=4,
        search_kwargs={"param": {"boost": {"title": 1.05}}},
    )

    # Pull the RAG prompt from the hub
    prompt = hub.pull("daethyra/rag-prompt")
    prompt_template = ChatPromptTemplate.from_template(prompt)
    output_parser = StrOutputParser()

    # Create a custom RAG chain
    rag_chain = (
        RunnableParallel({"context": formatted_docs, "question": RunnablePassthrough()})
        | prompt_template
        | chat_model
        | output_parser
    )

    # Get user query and invoke the RAG chain
    user_query = input("Please enter your query: ")
    result = rag_chain.invoke({"question": user_query})

    # Print the answer
    print(result)

except Exception as e:
    print(f"An error occurred: {e}")
