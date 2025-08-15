from app.utils.utils import convert_pdf_to_markdown, get_files_in_directory
from app.rag.collections import COLLECTIONS
from app.utils.chunking import chunking_document


def process_stockcode_pdf(input_dir):
    files = get_files_in_directory(input_dir)
    for f in files:
        if f.endswith(".pdf"):
            file_name = f.split("/")[-1][:-4]
            print(f"Processing file: {f}")
            markdown_content = convert_pdf_to_markdown(f)
            chunks = chunking_document(markdown_content)
            for chunk in chunks:  
                COLLECTIONS["stock"].insert_data(["content", "source"], [chunk, file_name], [0, 1])
