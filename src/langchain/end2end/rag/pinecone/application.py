from documentservice import DocumentService
from vectorstoreservice import VectorStoreService


class Application:
    """This class is responsible for running the application"""

    def __init__(self, data_directory, index_name):
        """Initialize the application with the given data directory and index name"""
        self.document_service = DocumentService(data_directory)
        self.vector_store_service = VectorStoreService(index_name)

    def run(self):
        """Run the application"""
        self.document_service.load_documents()  # Load documents from the data directory
        # Upsert documents into the vector store
        self.vector_store_service.upsert_documents(
            self.document_service.get_documents()
        )

        while True:
            query = input("Enter your query (or 'exit' to quit): ")  # Get user input
            if query.lower() == "exit":  # Check if user wants to exit
                break

            # Retrieve documents from the vector store
            results = self.vector_store_service.retrieve_documents(query)
            for result in results:
                print(result.page_content)  # Print the page content of each result
                print("-" * 80)  # Print a separator
