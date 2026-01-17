"""
RAG (Retrieval-Augmented Generation) Q&A
Tìm kiếm documents và tạo prompt cho LLM sử dụng CodeBERT
"""
import os
import sys
import io
import faiss
import pandas as pd
import numpy as np

# Import transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers không khả dụng.")

# Fix encoding removed to avoid Streamlit conflict

INDEX_PATH = 'data/processed/faiss_index.bin'
META_PATH = 'data/processed/metadf.parquet'
CODEBERT_MODEL = 'microsoft/codebert-base'

# Cache
_index_cache = None
_meta_cache = None
_tokenizer_cache = None
_model_cache = None

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_codebert():
    """Load CodeBERT model & tokenizer (Cached)"""
    global _tokenizer_cache, _model_cache
    
    if _tokenizer_cache is not None and _model_cache is not None:
        return _tokenizer_cache, _model_cache

    if not TRANSFORMERS_AVAILABLE:
         raise ImportError("Cần cài đặt transformers và torch để sử dụng CodeBERT.")

    print(f"Loading CodeBERT: {CODEBERT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    model = AutoModel.from_pretrained(CODEBERT_MODEL)
    
    device = get_device()
    model = model.to(device)
    model.eval()
    
    _tokenizer_cache = tokenizer
    _model_cache = model
    return tokenizer, model

def encode_query(query):
    """Encode query using CodeBERT with mean pooling"""
    tokenizer, model = load_codebert()
    device = get_device()
    
    # Encode
    inputs = tokenizer(
        [query], 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).float().cpu().numpy()
        
    return embedding[0]

def load_index(index_path=INDEX_PATH, meta_path=META_PATH, use_cache=True):
    """Load FAISS index và metadata"""
    global _index_cache, _meta_cache
    
    if use_cache and _index_cache is not None:
        return _index_cache, _meta_cache
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index không tồn tại tại {index_path}. Vui lòng chạy ingest_to_vectorstore.py trước.")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata không tồn tại tại {meta_path}. Vui lòng chạy ingest_to_vectorstore.py trước.")
    
    index = faiss.read_index(index_path)
    meta = pd.read_parquet(meta_path)
    
    if use_cache:
        _index_cache = index
        _meta_cache = meta
    
    return index, meta

def retrieve(query, k=5):
    """Tìm kiếm documents liên quan"""
    index, meta = load_index()
    
    # Encode query using CodeBERT
    q_emb = encode_query(query).astype('float32')
    
    # Search
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
    """Tạo prompt cho LLM"""
    ctx = ''
    for i, d in enumerate(docs, 1):
        ctx += f"Document {i} Title: {d['title']}\n{d['content']}\n---\n"
    
    prompt = f"""You are an assistant specialized in smart contract security. Use the following documents as context to answer the question.

Context:
{ctx}

Question: {query}

Answer concisely and accurately. Cite which document (number) you used for each part of your answer."""
    
    return prompt

def generate_answer_with_openai(query, docs, api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini"):
    """Generate answer bằng OpenAI API"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package chưa được cài đặt.")
    
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return "⚠️ Không tìm thấy API Key. Vui lòng thiết lập OPENAI_API_KEY trong .env hoặc nhập trong giao diện."
        
    client = OpenAI(api_key=api_key)
    prompt = compose_prompt(query, docs)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in smart contract security."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi OpenAI API: {str(e)}"

if __name__ == '__main__':
    q = "Common reentrancy attacks?"
    print(f"Query: {q}")
    docs = retrieve(q)
    print(f"Found {len(docs)} docs.")
    for d in docs:
        print(f"- {d['title']}")