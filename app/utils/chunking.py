from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd 
import re

def chunking_document(document, chunk_size=800, chunk_overlap=100):
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = spliter.split_text(document)
    return chunks

def split_sentences(text):
    # Tách theo dấu câu tiếng Việt (., !, ?) có thể kèm theo xuống dòng
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def extract_sector_sentences(documents, sector_keywords):
    matched_sentences = {}
    for document in documents:
        sentences = split_sentences(document)
        for sent in sentences:
            for sector, keywords in sector_keywords.items():
                for kw in keywords:
                    if kw.lower() in sent.lower():
                        if sector not in list(matched_sentences.keys()):
                            matched_sentences[sector] = [sent]
                        else:
                            matched_sentences[sector].append(sent) 
    return matched_sentences