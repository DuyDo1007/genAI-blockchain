"""
Tạo vector embeddings và FAISS index cho RAG
"""
import os
import sys
import io
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


CSV_PATH = 'data/processed/findings.csv'
EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_PATH = 'data/processed/faiss_index.bin'
META_PATH = 'data/processed/metadf.parquet'


def build_embeddings(csv_path=CSV_PATH, emb_model_name=EMB_MODEL):
    """
    Tạo embeddings và FAISS index từ CSV file
    
    Args:
        csv_path: Đường dẫn file CSV chứa findings
        emb_model_name: Tên embedding model
    
    Returns:
        FAISS index và metadata DataFrame
    """
    print("=" * 50)
    print("TẠO VECTOR STORE (FAISS INDEX)")
    print("=" * 50)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV tại {csv_path}. Vui lòng chạy data_preprocessing.py trước.")
    
    print(f"Đang đọc dữ liệu từ {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Đã load {len(df)} records")
    
    # Tạo text từ title và content
    df['text'] = (df['title'].fillna('') + '\n' + df['content'].fillna('')).astype(str)
    texts = df['text'].tolist()
    
    # Tạo embeddings
    print(f"Đang tạo embeddings bằng {emb_model_name}...")
    model = SentenceTransformer(emb_model_name)
    embeds = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"✓ Embeddings shape: {embeds.shape}")
    
    # Build FAISS index
    print("Đang tạo FAISS index...")
    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeds.astype('float32'))
    print(f"✓ FAISS index đã được tạo với {index.ntotal} vectors")
    
    # Lưu index và metadata
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(META_PATH, index=False)
    
    print(f"✓ Đã lưu index vào {INDEX_PATH}")
    print(f"✓ Đã lưu metadata vào {META_PATH}")
    
    return index, df


if __name__ == '__main__':
    build_embeddings()