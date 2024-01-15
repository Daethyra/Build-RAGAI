"""Module for retrieving documents with integrated multi-vector and multi-query capabilities."""

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever


# Build a function that integrates `MultiVectorRetriever` & `MultiQueryRetriever`
def multi_vector_query_retriever(question):
    """
    Retrieve relevant documents from multiple perspectives based on a given question.

    This function combines the power of MultiVectorRetriever and MultiQueryRetriever to enhance the retrieval process.
    It generates multiple queries using MultiVectorRetriever and retrieves relevant documents for each query using MultiQueryRetriever.

    Parameters:
        question (str): The question or query for which relevant documents are to be retrieved.

    Returns:
        List[Document]: A list of relevant documents retrieved from multiple perspectives.
          --> Create your own formatting function based on your use-case
    """

    # Create the MultiVectorRetriever
    vector_retriever = MultiVectorRetriever()

    # Generate multiple queries using the MultiVectorRetriever
    queries = vector_retriever.generate_queries(question)

    # Create the MultiQueryRetriever
    query_retriever = MultiQueryRetriever()

    # Get relevant documents for each query
    relevant_documents = []
    for query in queries:
        documents = query_retriever.get_relevant_documents(query)
        relevant_documents.extend(documents)

    return relevant_documents


# Example function call
# relevant_documents = multi_vector_query_retriever(question)
