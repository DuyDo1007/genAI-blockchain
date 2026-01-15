"""
RAG (Retrieval-Augmented Generation) Q&A
Tìm kiếm documents và tạo prompt cho LLM
"""
import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np


INDEX_PATH = 'data/processed/faiss_index.bin'
META_PATH = 'data/processed/metadf.parquet'
EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


# Cache để tránh load lại nhiều lần
_index_cache = None
_meta_cache = None
_model_cache = None


def load_index(index_path=INDEX_PATH, meta_path=META_PATH, use_cache=True):
    """
    Load FAISS index và metadata
    
    Args:
        index_path: Đường dẫn FAISS index
        meta_path: Đường dẫn metadata parquet
        use_cache: Sử dụng cache nếu True
    
    Returns:
        (index, meta, model) tuple
    """
    global _index_cache, _meta_cache, _model_cache
    
    if use_cache and _index_cache is not None:
        return _index_cache, _meta_cache, _model_cache
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index không tồn tại tại {index_path}. Vui lòng chạy ingest_to_vectorstore.py trước.")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata không tồn tại tại {meta_path}. Vui lòng chạy ingest_to_vectorstore.py trước.")
    
    index = faiss.read_index(index_path)
    meta = pd.read_parquet(meta_path)
    model = SentenceTransformer(EMB_MODEL)
    
    if use_cache:
        _index_cache = index
        _meta_cache = meta
        _model_cache = model
    
    return index, meta, model


def retrieve(query, k=5):
    """
    Tìm kiếm documents liên quan đến query
    
    Args:
        query: Câu hỏi hoặc từ khóa tìm kiếm
        k: Số lượng documents cần trả về
    
    Returns:
        List[Dict]: Danh sách documents với id, title, content
    """
    index, meta, model = load_index()
    
    # Encode query
    q_emb = model.encode([query])[0].astype('float32')
    
    # Tìm kiếm trong FAISS
    D, I = index.search(np.array([q_emb]), k)
    
    # Lấy documents
    docs = []
    for idx in I[0]:
        if idx < 0 or idx >= len(meta):
            continue
        row = meta.iloc[idx]
        docs.append({
            'id': int(row['id']),
            'title': str(row['title']),
            'content': str(row['content'])
        })
    
    return docs


def compose_prompt(query, docs):
    """
    Tạo prompt cho LLM từ query và documents
    
    Args:
        query: Câu hỏi
        docs: Danh sách documents đã retrieve
    
    Returns:
        str: Prompt đầy đủ cho LLM
    """
    ctx = ''
    for i, d in enumerate(docs, 1):
        ctx += f"Document {i} Title: {d['title']}\n{d['content']}\n---\n"
    
    prompt = f"""You are an assistant specialized in smart contract security. Use the following documents as context to answer the question.

Context:
{ctx}

Question: {query}

Answer concisely and accurately. Cite which document (number) you used for each part of your answer."""
    
    return prompt


def generate_answer_with_openai(query, docs, api_key=None, model="gpt-3.5-turbo"):
    """
    Generate answer bằng OpenAI API
    
    Args:
        query: Câu hỏi
        docs: Documents đã retrieve
        api_key: OpenAI API key
        model: Model name
    
    Returns:
        str: Generated answer
    """
    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI package chưa được cài đặt. Cần: pip install openai")
    
    if api_key:
        openai.api_key = api_key
    elif not os.getenv('OPENAI_API_KEY'):
        raise ValueError("Cần cung cấp OpenAI API key")
    
    prompt = compose_prompt(query, docs)
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in smart contract security."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content


if __name__ == '__main__':
    q = "What are common causes of reentrancy?"
    print(f"Query: {q}\n")
    
    docs = retrieve(q, k=3)
    print(f"Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc['title']} (ID: {doc['id']})")
    
    prompt = compose_prompt(q, docs)
    print(f"\n--- Prompt ---\n{prompt[:500]}...")