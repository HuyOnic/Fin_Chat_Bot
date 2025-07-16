def build_prompt(user_query:str, documents: list[dict]):
    context = "\n---\n".join(doc["content"] for doc in documents)
    prompt = f"Trả lời câu hỏi: {user_query} bằng cách dựa vào tin tức\n\nTin tức:\n{context}\n\nTrả lời chi tiết bằng Tiếng Việt, không sử dụng Tiếng Anh:"
    return prompt

def basic_system_prompt(documents: list[dict]):
    content = "\n---\n".join(doc["content"] for doc in documents)
    source = "\n---\n".join(doc["source"] for doc in documents)
    prompt = f"""Bạn là một trợ lý AI chuyên về tài chính, hãy trở lời chính xác dựa vào tài liệu sau

    Tài liệu:
    
    {content}
    
    Trả lời chi tiết cho câu hỏi của người dùng dưới đây bằng Tiếng Việt, không sử dụng Tiếng Anh.
    """
    return (prompt)

ANSWER_FINANCIAL_QUESTION_PROMPT = """Bạn là một trợ lý AI chuyên về tài chính, hãy trở lời chính xác dựa vào tài liệu sau

Tài liệu:

{context}

Trả lời chi tiết cho câu hỏi của người dùng dưới đây bằng Tiếng Việt, không sử dụng Tiếng Anh.

Câu hỏi: {question}
"""
