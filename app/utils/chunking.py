from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd 

def chunking_document(document, chunk_size=800, chunk_overlap=100):
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = spliter.split_text(document)
    return chunks