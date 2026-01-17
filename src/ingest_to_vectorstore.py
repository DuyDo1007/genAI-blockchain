"""
Tạo vector embeddings và FAISS index cho RAG sử dụng CodeBERT
"""
import os
import sys
import io
import pandas as pd
import faiss
import numpy as np

# Import transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers không khả dụng.")

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

CSV_PATH = 'data/processed/findings.csv'
INDEX_PATH = 'data/processed/faiss_index.bin'
META_PATH = 'data/processed/metadf.parquet'
CODEBERT_MODEL = 'microsoft/codebert-base'

def create_embeddings(texts, model_name=CODEBERT_MODEL, batch_size=16, max_length=512):
    """
    Tạo embeddings sử dụng CodeBERT
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers change required for CodeBERT.")
    
    print(f"Đang load CodeBERT model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = model(**encoded)
                # Mean pooling
                input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).float().cpu().numpy()
            
            embeddings.append(batch_embeddings)
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Đã xử lý {i + len(batch_texts)}/{len(texts)} samples...")
        except Exception as e:
            print(f"Lỗi batch {i}: {e}")
            continue
    
    if not embeddings: return np.array([])
    return np.vstack(embeddings)

def build_embeddings(csv_path=CSV_PATH):
    """
    Tạo embeddings và FAISS index từ CSV file
    """
    print("=" * 50)
    print("TẠO VECTOR STORE (FAISS INDEX) VỚI CODEBERT")
    print("=" * 50)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV tại {csv_path}. Vui lòng chạy data_preprocessing.py trước.")
    
    print(f"Đang đọc dữ liệu từ {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Đã load {len(df)} records")
    
    # Tạo text từ title và content
    df['text'] = (df['title'].fillna('') + '\n' + df['content'].fillna('')).astype(str)
    # Lấy mẫu nhỏ hơn để tránh OOM nếu cần, nhưng CodeBERT cũng handle được 512 tokens
    texts = df['text'].tolist()
    
    # Tạo embeddings
    print(f"Đang tạo embeddings bằng {CODEBERT_MODEL}...")
    embeds = create_embeddings(texts)
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