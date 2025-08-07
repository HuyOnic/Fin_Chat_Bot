import os
from app.utils.utils import convert_pdf_to_markdown, get_files_in_directory
from app.db.qdrant import insert_vector, STOCKCODE_COLLECTION_NAME
from app.utils.vectorizer import convert_to_dense_vector, convert_to_sparse_vector
from app.utils.chunking import chunking_document
import uuid


def process_stockcode_pdf(input_dir):
    files = get_files_in_directory(input_dir)
    for f in files:
        if f.endswith(".pdf"):
            file_name = f.split("/")[-1][:-4]
            print(f"Processing file: {f}")
            markdown_content = convert_pdf_to_markdown(f)
            chunks = chunking_document(markdown_content)
            for chunk in chunks:
                dense_vector = convert_to_dense_vector([f"source: {file_name} \n {chunk}"])[0]
                sparse_vector = convert_to_sparse_vector(f"source: {file_name} \n {chunk}")
                chunk_id = str(uuid.uuid4())
                insert_vector(
                    article_id=chunk_id,
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,    
                    payload_keys=["content", "source"],
                    payload_values=[chunk, file_name],
                    collection_name=STOCKCODE_COLLECTION_NAME
                )         
