import os
from docling.document_converter import DocumentConverter

def convert_pdf_to_markdown(path):
    converter = DocumentConverter()
    result = converter.convert(path)
    return result.document.export_to_markdown()


def get_files_in_directory(path):
    files = []
    for f in os.listdir(path):
        path_f = os.path.join(path, f)
        if os.path.isfile(path_f):
            files.append(path_f)
        elif os.path.isdir(path_f):
            files.extend(get_files_in_directory(path_f))

    return files