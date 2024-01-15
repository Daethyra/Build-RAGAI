""" This module defines the Application class responsible for running the entire application.
"""

from documentservice import DocumentService
from vectorstoreservice import VectorStoreService


class Application:

    def __init__(self, data_directory, index_name):
        """
        Initializes a new instance of the class.

        Args:
            data_directory (str): The directory where the data is stored.
            index_name (str): The name of the index.

        Returns:
            None
        """

        self.document_service = DocumentService(data_directory)
        self.vector_store_service = VectorStoreService(index_name)

    def run(self):
        """
        Run the program.

        This function executes the main logic of the program. It performs the following steps:
        1. Load documents from the data directory.
        2. Upsert documents into the vector store.
        3. Prompt the user for a query.
        4. Retrieve documents from the vector store based on the query.
        5. Print the page content of each retrieved document.

        Parameters:
        - None

        Returns:
        - None
        """
        # Load documents from the data directory
        self.document_service.load_documents()
        
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
