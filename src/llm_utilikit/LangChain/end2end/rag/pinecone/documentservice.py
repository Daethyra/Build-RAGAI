from langchain_community.document_loaders import UnstructuredFileLoader


class DocumentService:
    def __init__(self, data_directory):
        self.loader = UnstructuredFileLoader(data_directory)
        self.documents = []

    def load_documents(self):
        self.documents = self.loader.load()

    def get_documents(self):
        return self.documents
