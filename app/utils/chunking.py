from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd 
import re

def chunking_document(document, chunk_size=1024, chunk_overlap=256):
    spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    chunks = spliter.split_text(document)
    chunks = [chunk for chunk in chunks if len(chunk) >= 40]
    return chunks

def split_sentences(text):
    # Tách theo dấu câu tiếng Việt (., !, ?) có thể kèm theo xuống dòng
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# def extract_sector_sentences(news, sector_keywords):
#     matched_sentences = {}
#     for item in news:
#         sentences = split_sentences(item["content"])
#         for sent in sentences:
#             for sector, keywords in sector_keywords.items():
#                 for kw in keywords:
#                     if kw.lower() in sent.lower():
#                         if sector not in list(matched_sentences.keys()):
#                             matched_sentences[sector] = {"sentence":[sent],
#                                                          "source":[item["source"]]}
#                         else:
#                             matched_sentences[sector]["sentence"].append(sent)
#                             matched_sentences[sector]["source"].append(item["source"])  
#     return matched_sentences

def extract_sector_sentences(news, sector_keywords):
    matched_sentences = {}

    for item in news:
        sentences = split_sentences(item["content"])
        source = item["source"]
        
        for sent in sentences:
            sent_lower = sent.lower()
            for sector, keywords in sector_keywords.items():
                if any(kw.lower() in sent_lower for kw in keywords):
                    matched = matched_sentences.setdefault(sector, {"sentence": [], "source": []})
                    matched["sentence"].append(sent)
                    matched["source"].append(source)
    return matched_sentences
