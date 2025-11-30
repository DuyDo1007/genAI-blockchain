import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_PATH = 'data/processed/faiss_index.bin'
META_PATH = 'data/processed/metadf.parquet'
EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

_index_cache = None
_meta_cache = None
_model_cache = None


def load_index(index_path=INDEX_PATH, meta_path=META_PATH, use_cache=True):
    global _index_cache, _meta_cache, _model_cache
    
    if use_cache and _index_cache is not None:
        return _index_cache, _meta_cache, _model_cache
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Run ingest_to_vectorstore.py first.")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}. Run ingest_to_vectorstore.py first.")
    
    index = faiss.read_index(index_path)
    meta = pd.read_parquet(meta_path)
    model = SentenceTransformer(EMB_MODEL)
    
    if use_cache:
        _index_cache = index
        _meta_cache = meta
        _model_cache = model
    
    return index, meta, model


def retrieve(query, k=5):
    index, meta, model = load_index()
    q_emb = model.encode([query])[0].astype('float32')
    D, I = index.search(np.array([q_emb]), k)
    
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
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Provide as parameter or set OPENAI_API_KEY env var.")
        client = OpenAI(api_key=api_key)
    
    prompt = compose_prompt(query, docs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in smart contract security."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content if response.choices else ""


def rag_query(query, api_key=None, k=5, model="gpt-3.5-turbo"):
    """Full RAG pipeline: retrieve + generate answer"""
    docs = retrieve(query, k=k)
    answer = generate_answer_with_openai(query, docs, api_key=api_key, model=model)
    return {'query': query, 'documents': docs, 'answer': answer}


if __name__ == '__main__':
    q = "What are common causes of reentrancy?"
    print(f"Query: {q}\n")
    
    docs = retrieve(q, k=3)
    print(f"Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc['title']} (ID: {doc['id']})")
    
    prompt = compose_prompt(q, docs)
    print(f"\n--- Prompt ---\n{prompt[:500]}...")