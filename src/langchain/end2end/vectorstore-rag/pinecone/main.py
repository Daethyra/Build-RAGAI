""" This module is the main entry point of the application. """

from application import Application

if __name__ == "__main__":
    data_directory = "data"
    index_name = "rag-testing-smiley-face"

    app = Application(data_directory, index_name)
    app.run()
