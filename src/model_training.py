"""
Huấn luyện mô hình phát hiện lỗ hổng Smart Contract
Sử dụng CodeBERT và Supervised Learning (RandomForest)
"""
import os
import sys
import io
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers không khả dụng. Vui lòng cài đặt: pip install transformers torch")

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

META_PATH = 'data/processed/metadf.parquet'
CSV_PATH = 'data/processed/findings.csv'
MODEL_PATH_CLF = 'models/trained_classifier.pkl'

# CodeBERT configuration
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

def create_embeddings(texts, model_name=CODEBERT_MODEL, batch_size=8, max_length=512):
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
    print(f"  - Sử dụng device: {device}")
    
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
                # Mean pooling (taking attention mask into account)
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
            
    if not embeddings:
        return np.array([])
        
    return np.vstack(embeddings)

def extract_code_features(df):
    """Trích xuất features từ code (Utility shared)"""
    features = {}
    if 'code' not in df.columns: return features
    code_col = df['code'].astype(str)
    
    features['code_length'] = code_col.str.len()
    features['num_lines'] = code_col.str.count('\n') + 1
    features['num_functions'] = code_col.str.lower().str.count('function')
    
    # Security-related patterns
    features['has_require'] = code_col.str.contains(r'\brequire\b', case=False, na=False)
    features['num_require'] = code_col.str.lower().str.count('require')
    
    # Vulnerability indicators
    features['has_transfer'] = code_col.str.contains(r'\.transfer\(|transfer\(', case=False, na=False)
    features['has_call'] = code_col.str.contains(r'\.call\(|call\(', case=False, na=False)
    features['has_delegatecall'] = code_col.str.contains(r'delegatecall', case=False, na=False)
    
    return features

def create_ground_truth_labels(df):
    """Tạo labels cho bài toán supervised learning"""
    labels = np.zeros(len(df))
    
    # 1. Impact based
    high_impact_mask = df['impact'].str.upper() == 'HIGH'
    labels[high_impact_mask] = 1
    
    medium_impact_mask = df['impact'].str.upper() == 'MEDIUM'
    if 'vulnerability_label' in df.columns:
        critical_vulns = ['REENTRANCY', 'OVERFLOW', 'UNDERFLOW', 'ACCESS_CONTROL']
        for vuln in critical_vulns:
            vuln_mask = df['vulnerability_label'].str.upper().str.contains(vuln, na=False)
            labels[medium_impact_mask & vuln_mask] = 1
            
    # 2. Code features based heuristic (weak labeling)
    if 'code' in df.columns:
        code_features = extract_code_features(df)
        if 'code_length' in code_features:
            long_code = code_features['code_length'] > code_features['code_length'].quantile(0.75)
            # Simple heuristic
            has_vuln_patterns = (
                code_features.get('has_transfer', pd.Series([False]*len(df))) |
                code_features.get('has_call', pd.Series([False]*len(df))) |
                code_features.get('has_delegatecall', pd.Series([False]*len(df)))
            )
            labels[long_code & has_vuln_patterns] = 1
            
    return labels

def train_classifier(n_estimators=100, random_state=42):
    """Huấn luyện Supervised Classifier (RandomForest) với CodeBERT"""
    print("=" * 50)
    print("Huấn luyện Classifier (Supervised) với CodeBERT")
    print("=" * 50)
    
    df = load_data()
    # Lọc chỉ lấy dòng có code
    df_with_code = df[df['code'].notna() & (df['code'].astype(str).str.strip() != '')]
    
    if len(df_with_code) == 0:
        raise ValueError("Không tìm thấy dữ liệu code hợp lệ.")
        
    print(f"  - Dữ liệu: {len(df_with_code)} samples")
    
    # Prepare Features (Embeddings)
    texts = df_with_code['code'].astype(str).tolist()
    print("Đang tạo CodeBERT embeddings...")
    emb = create_embeddings(texts)
    
    if emb.size == 0:
         raise ValueError("Không tạo được embeddings.")

    # Prepare Labels
    y = create_ground_truth_labels(df_with_code)
    print(f"  - Original Labels distribution: Safe={np.sum(y==0)}, Vuln={np.sum(y==1)}")
    
    # --- DATA BALANCING (Oversampling) ---
    # Convert to dataframe for easy resampling
    df_train = pd.DataFrame({'emb_idx': range(len(emb)), 'label': y})
    
    # Separate classes
    safe_indices = df_train[df_train['label'] == 0].index
    vuln_indices = df_train[df_train['label'] == 1].index
    
    # Resample Vuln to match Safe count
    if len(vuln_indices) < len(safe_indices):
        print(f"  - Oversampling Vuln class from {len(vuln_indices)} to {len(safe_indices)} samples...")
        vuln_indices_oversampled = np.random.choice(vuln_indices, size=len(safe_indices), replace=True)
        balanced_indices = np.concatenate([safe_indices, vuln_indices_oversampled])
    else:
        balanced_indices = df_train.index
        
    # Get balanced data
    emb_balanced = emb[balanced_indices]
    y_balanced = y[balanced_indices]
    print(f"  - Balanced Labels distribution: Safe={np.sum(y_balanced==0)}, Vuln={np.sum(y_balanced==1)}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(emb_balanced, y_balanced, test_size=0.2, random_state=random_state)
    
    # Train
    print(f"Đang train RandomForest ({n_estimators} estimators)...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1) # Removed class_weight as we oversampled
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("Kết quả trên tập test split:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Vuln']))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    meta = {
        'clf': clf,
        'emb_model_name': CODEBERT_MODEL,
        'use_codebert': True,
        'type': 'classifier'
    }
    joblib.dump(meta, MODEL_PATH_CLF)
    print(f"✓ Đã lưu Classifier vào {MODEL_PATH_CLF}")
    return meta

if __name__ == '__main__':
    train_classifier()