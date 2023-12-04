import logging
from qa_local_docs import PDFProcessor


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    # Set up logging
    setup_logging()

    try:
        # Initialize PDFProcessor class
        pdf_processor = PDFProcessor()

        # Load PDFs from directory and count the number of loaded documents
        texts = pdf_processor.load_pdfs_from_directory()
        num_docs = len(texts)
        logging.info(f"Loaded {num_docs} document(s) from directory.")

        # Perform similarity search based on the query
        query = pdf_processor.get_user_query()
        logging.debug(f"User query: {query}")
        results = pdf_processor.perform_similarity_search(texts, query)

        # Log the results
        if results:
            logging.info(f"Found {len(results)} similar document(s) for query: {query}")
            for i, result in enumerate(results):
                logging.debug(
                    f"{i+1}. Similarity score: {result['similarity_score']}, \nDocument: {result['document']}"
                )
        else:
            logging.warning(f"No similar documents found for query: {query}")

        # Answer a question using the RAG model
        question = pdf_processor.get_user_query(
            """Welcome! \
            \nYour document agent has been fully instantiated. \ 
            Please enter a clear and concise question: """
        )
        logging.debug(f"User question: {question}")
        answer = pdf_processor.answer_question(question)
        logging.info(f"\nAnswer: {answer}")
    except FileNotFoundError as fe:
        logging.error(f"FileNotFoundError encountered: {fe}")
    except ValueError as ve:
        logging.error(f"ValueError encountered: {ve}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
