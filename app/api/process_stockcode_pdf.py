import os
from app.utils.utils import convert_pdf_to_markdown, get_files_in_directory
from app.db.qdrant import insert_vector, STOCKCODE_COLLECTION_NAME
from app.utils.vectorizer import convert_to_vector
from app.utils.chunking import chunking_document
import uuid


def process_stockcode_pdf(input_dir):
    files = get_files_in_directory(input_dir)
    for f in files:
        if f.endswith(".pdf"):
            print(f"Processing file: {f}")
            markdown_content = convert_pdf_to_markdown(f)
            chunks = chunking_document(markdown_content)
            for chunk in chunks:
                vector = convert_to_vector([chunk])[0]       
                chunk_id = str(uuid.uuid4())
                insert_vector(
                    article_id=chunk_id,
                    vector=vector,
                    payload_keys=["content"],
                    payload_values=[chunk],
                    collection_name=STOCKCODE_COLLECTION_NAME
                )         

                