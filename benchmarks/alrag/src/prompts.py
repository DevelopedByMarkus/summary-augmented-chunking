# PROMPT TEMPLATE FOR RAG MODE
RAG_PROMPT_TEMPLATE = """You are an expert legal AI assistant. Your task is to answer the following question based on the provided context snippets from a legal document.
Provide a direct and concise answer.

[Context Snippets Start]
{context}
[Context Snippets End]

[Question Start]
{question}
[Question End]

Answer:
"""

# PROMPT TEMPLATE FOR NO-RAG (BASELINE) MODE
NO_RAG_PROMPT_TEMPLATE = """You are an expert legal AI assistant. Your task is to answer the following question based on your internal knowledge.
Provide a direct and concise answer.

[Question Start]
{question}
[Question End]

Answer:
"""
