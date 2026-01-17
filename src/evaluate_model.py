"""
Đánh giá mô hình phát hiện lỗ hổng Smart Contract
Sử dụng CodeBERT và Supervised Learning
"""
import os
import sys
import io
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

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

MODEL_PATH_CLF = 'models/trained_classifier.pkl'
META_PATH = 'data/processed/metadf.parquet'
CSV_PATH = 'data/processed/findings.csv'
CODEBERT_MODEL = 'microsoft/codebert-base'

def load_data():
    """Load dữ liệu từ CSV hoặc Parquet"""
    if os.path.exists(META_PATH):
        df = pd.read_parquet(META_PATH)
    elif os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tại {META_PATH} hoặc {CSV_PATH}")
    return df

def create_embeddings(texts, model_name=CODEBERT_MODEL, batch_size=16, max_length=512):
    """
    Tạo embeddings sử dụng CodeBERT (Replicated logic for standalone execution)
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

def extract_code_features(df):
    """Trích xuất features từ code (Shared utility)"""
    features = {}
    if 'code' not in df.columns: return features
    code_col = df['code'].astype(str)
    
    features['code_length'] = code_col.str.len()
    features['num_lines'] = code_col.str.count('\n') + 1
    features['num_functions'] = code_col.str.lower().str.count('function')
    features['has_require'] = code_col.str.contains(r'\brequire\b', case=False, na=False)
    features['num_require'] = code_col.str.lower().str.count('require')
    features['has_transfer'] = code_col.str.contains(r'\.transfer\(|transfer\(', case=False, na=False)
    
    return features

def create_ground_truth_labels(df):
    """Tạo labels cho bài toán supervised learning"""
    labels = np.zeros(len(df))
    
    high_impact_mask = df['impact'].str.upper() == 'HIGH'
    labels[high_impact_mask] = 1
    
    medium_impact_mask = df['impact'].str.upper() == 'MEDIUM'
    if 'vulnerability_label' in df.columns:
        critical_vulns = ['REENTRANCY', 'OVERFLOW', 'UNDERFLOW', 'ACCESS_CONTROL']
        for vuln in critical_vulns:
            vuln_mask = df['vulnerability_label'].str.upper().str.contains(vuln, na=False)
            labels[medium_impact_mask & vuln_mask] = 1
            
    if 'code' in df.columns:
        code_features = extract_code_features(df)
        if 'code_length' in code_features:
            long_code = code_features['code_length'] > code_features['code_length'].quantile(0.75)
            # Heuristic
            has_vuln_patterns = code_features.get('has_transfer', pd.Series([False]*len(df)))
            labels[long_code & has_vuln_patterns] = 1
            
    return labels

def evaluate_classifier():
    """Đánh giá Supervised Classifier"""
    print("=" * 50)
    print("DANH GIA SUPERVISED CLASSIFIER (CodeBERT)")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH_CLF):
        raise FileNotFoundError(f"Model không tồn tại tại {MODEL_PATH_CLF}")
        
    print("Đang load model...")
    meta = joblib.load(MODEL_PATH_CLF)
    clf = meta['clf']
    
    print("Đang load dữ liệu...")
    df = load_data()
    df_with_code = df[df['code'].notna() & (df['code'].astype(str).str.strip() != '')]
    
    print("Đang tạo embeddings...")
    texts = df_with_code['code'].astype(str).tolist()
    embeddings = create_embeddings(texts)
    
    print("Đang predict...")
    y_probs = clf.predict_proba(embeddings)[:, 1]
    # Threshold adjustment for Evaluation
    THRESHOLD = 0.3
    y_pred = (y_probs > THRESHOLD).astype(int)
    
    y_true = create_ground_truth_labels(df_with_code)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "-" * 50)
    print("METRICS (Classifier)")
    print("-" * 50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    
    print(classification_report(y_true, y_pred, target_names=['Safe', 'Vuln']))
    
    return {'f1': f1}

if __name__ == '__main__':
    evaluate_classifier()
