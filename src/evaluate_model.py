"""
Đánh giá mô hình phát hiện bất thường (Anomaly Detection)
Tính Precision, Recall, F1 score
"""
import os
import sys
import io
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Thử import transformers cho CodeBERT
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_PATH_IF = 'models/trained_if.pkl'
MODEL_PATH_AE = 'models/autoencoder.h5'
META_PATH = 'data/processed/metadf.parquet'
CSV_PATH = 'data/processed/findings.csv'
EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
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


def create_embeddings_codebert(texts, model_name=CODEBERT_MODEL, batch_size=16, max_length=512):
    """Tạo embeddings từ CodeBERT model"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers không khả dụng. Cần cài: pip install transformers torch")
    
    print(f"Đang load CodeBERT model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  - Sử dụng device: {device}")
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
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
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        embeddings.append(batch_embeddings)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Đã xử lý {i + len(batch_texts)}/{len(texts)} samples...")
    
    return np.vstack(embeddings)


def create_embeddings(texts, use_codebert=False, model_name=None):
    """Tạo embeddings từ text, hỗ trợ cả SentenceTransformer và CodeBERT"""
    if use_codebert and TRANSFORMERS_AVAILABLE:
        if model_name is None:
            model_name = CODEBERT_MODEL
        print(f"✓ Sử dụng CodeBERT: {model_name}")
        return create_embeddings_codebert(texts, model_name)
    else:
        if model_name is None:
            model_name = EMB_MODEL
        print(f"✓ Sử dụng SentenceTransformer: {model_name}")
        model = SentenceTransformer(model_name)
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def extract_code_features(df):
    """
    Trích xuất features từ code để cải thiện ground truth và model
    """
    features = {}
    
    # Đảm bảo có cột code
    if 'code' not in df.columns:
        return features
    
    code_col = df['code'].astype(str)
    
    # Basic code metrics
    features['code_length'] = code_col.str.len()
    features['num_lines'] = code_col.str.count('\n') + 1
    # Sử dụng contains và sum thay vì count với regex
    features['num_functions'] = code_col.str.lower().str.count('function')
    features['num_contracts'] = code_col.str.lower().str.count('contract')
    
    # Security-related patterns
    features['has_require'] = code_col.str.contains(r'\brequire\b', case=False, na=False)
    features['has_assert'] = code_col.str.contains(r'\bassert\b', case=False, na=False)
    features['has_revert'] = code_col.str.contains(r'\brevert\b', case=False, na=False)
    features['num_require'] = code_col.str.lower().str.count('require')
    
    # Vulnerability indicators
    features['has_transfer'] = code_col.str.contains(r'\.transfer\(|transfer\(', case=False, na=False)
    features['has_send'] = code_col.str.contains(r'\.send\(|send\(', case=False, na=False)
    features['has_call'] = code_col.str.contains(r'\.call\(|call\(', case=False, na=False)
    features['has_delegatecall'] = code_col.str.contains(r'delegatecall', case=False, na=False)
    features['has_loop'] = code_col.str.contains(r'\bfor\b|\bwhile\b', case=False, na=False)
    features['has_condition'] = code_col.str.contains(r'\bif\b|\brequire\b|\bassert\b', case=False, na=False)
    
    # Critical vulnerability keywords
    vuln_keywords = [
        'reentrancy', 'overflow', 'underflow', 'access control',
        'unchecked', 'uninitialized', 'front-running', 'timestamp',
        'reentrant', 'race condition', 'denial of service', 'dos'
    ]
    for keyword in vuln_keywords:
        features[f'has_{keyword.replace(" ", "_")}'] = code_col.str.contains(keyword, case=False, na=False)
    
    return features


def create_ground_truth_labels(df):
    """
    Tạo ground truth labels cải thiện dựa trên:
    - Impact level
    - Vulnerability types
    - Code features và patterns
    """
    labels = np.zeros(len(df))
    
    # 1. HIGH impact = anomaly
    high_impact_mask = df['impact'].str.upper() == 'HIGH'
    labels[high_impact_mask] = 1
    
    # 2. MEDIUM impact với critical vulnerabilities = anomaly
    medium_impact_mask = df['impact'].str.upper() == 'MEDIUM'
    if 'vulnerability_label' in df.columns:
        critical_vulns = [
            'REENTRANCY', 'OVERFLOW', 'UNDERFLOW', 'ACCESS_CONTROL', 
            'UNCHECKED_CALL', 'UNINITIALIZED', 'FRONT_RUNNING'
        ]
        for vuln in critical_vulns:
            vuln_mask = df['vulnerability_label'].str.upper().str.contains(vuln, na=False)
            labels[medium_impact_mask & vuln_mask] = 1
    
    # 3. Code-based features (nếu có code)
    if 'code' in df.columns:
        code_features = extract_code_features(df)
        
        # Code có độ phức tạp cao và có vulnerability patterns
        if 'code_length' in code_features:
            long_code = code_features['code_length'] > code_features['code_length'].quantile(0.75)
            has_vuln_patterns = (
                code_features.get('has_transfer', pd.Series([False] * len(df))) |
                code_features.get('has_call', pd.Series([False] * len(df))) |
                code_features.get('has_delegatecall', pd.Series([False] * len(df))) |
                code_features.get('has_reentrancy', pd.Series([False] * len(df)))
            )
            # Code dài + có patterns nguy hiểm = có thể là anomaly
            labels[long_code & has_vuln_patterns] = 1
        
        # Code có nhiều functions nhưng ít require/assert = có thể không an toàn
        if 'num_functions' in code_features and 'num_require' in code_features:
            many_functions = code_features['num_functions'] > 3
            few_checks = code_features['num_require'] < 2
            labels[many_functions & few_checks] = 1
    
    # Convert: 1 = anomaly (positive), 0 = normal (negative)
    # IsolationForest: -1 = anomaly, 1 = normal
    y_true = np.where(labels == 1, -1, 1)
    
    return y_true, labels


def evaluate_isolation_forest():
    """Đánh giá IsolationForest model"""
    print("=" * 50)
    print("DANH GIA ISOLATION FOREST MODEL")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH_IF):
        raise FileNotFoundError(f"Model không tồn tại tại {MODEL_PATH_IF}. Vui lòng train model trước.")
    
    print("Đang load model...")
    meta = joblib.load(MODEL_PATH_IF)
    clf = meta['clf']
    emb_model_name = meta.get('emb_model_name', EMB_MODEL)
    use_codebert = meta.get('use_codebert', False)
    
    print("Đang load dữ liệu...")
    df = load_data()
    
    # Lọc chỉ lấy các dòng có code (giống như training)
    df_with_code = df[df['code'].notna() & (df['code'].astype(str).str.strip() != '')]
    if len(df_with_code) == 0:
        raise ValueError("Không có dòng nào có code trong dữ liệu!")
    
    print(f"  - Tổng số dòng: {len(df)}")
    print(f"  - Dòng có code: {len(df_with_code)}")
    
    print("Đang tạo embeddings từ CODE (giống như training)...")
    texts = df_with_code['code'].astype(str).tolist()
    embeddings = create_embeddings(texts, use_codebert=use_codebert, model_name=emb_model_name)
    
    print("Đang predict...")
    predictions = clf.predict(embeddings)
    scores = clf.decision_function(embeddings)
    
    # Tạo ground truth labels (sử dụng df_with_code thay vì df)
    y_true, labels_binary = create_ground_truth_labels(df_with_code)
    
    # Tính metrics
    # Convert predictions: -1 -> 1 (anomaly), 1 -> 0 (normal) cho tính toán metrics
    y_pred_binary = np.where(predictions == -1, 1, 0)
    y_true_binary = labels_binary
    
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    print("\n" + "=" * 50)
    print("KET QUA DANH GIA")
    print("=" * 50)
    print(f"Tổng số samples: {len(predictions)}")
    print(f"Anomalies phát hiện: {np.sum(predictions == -1)} ({np.sum(predictions == -1)/len(predictions)*100:.2f}%)")
    print(f"Normal samples: {np.sum(predictions == 1)} ({np.sum(predictions == 1)/len(predictions)*100:.2f}%)")
    print(f"\nGround Truth:")
    print(f"  Anomalies thực tế: {np.sum(y_true_binary == 1)} ({np.sum(y_true_binary == 1)/len(y_true_binary)*100:.2f}%)")
    print(f"  Normal thực tế: {np.sum(y_true_binary == 0)} ({np.sum(y_true_binary == 0)/len(y_true_binary)*100:.2f}%)")
    
    print("\n" + "-" * 50)
    print("METRICS")
    print("-" * 50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    print("\n" + "-" * 50)
    print("CONFUSION MATRIX")
    print("-" * 50)
    print("                Predicted")
    print("              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"       Anomaly  {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    # Classification Report
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(y_true_binary, y_pred_binary, 
                                target_names=['Normal', 'Anomaly'],
                                zero_division=0))
    
    # Score statistics
    print("\n" + "-" * 50)
    print("SCORE STATISTICS")
    print("-" * 50)
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std:  {scores.std():.4f}")
    print(f"Min:  {scores.min():.4f}")
    print(f"Max:  {scores.max():.4f}")
    
    # Top anomalies
    print("\n" + "=" * 50)
    print("TOP 10 MOST ANOMALOUS SAMPLES")
    print("=" * 50)
    top_anomalies_idx = np.argsort(scores)[:10]
    for i, idx in enumerate(top_anomalies_idx, 1):
        row = df_with_code.iloc[idx]
        is_actual_anomaly = "✓" if y_true_binary[idx] == 1 else "✗"
        print(f"\n{i}. ID: {row['id']}, Score: {scores[idx]:.4f} {is_actual_anomaly}")
        print(f"   Title: {row['title'][:80]}...")
        print(f"   Impact: {row.get('impact', 'N/A')}")
        print(f"   Code length: {len(str(row.get('code', '')))} chars")
    
    # Lưu kết quả
    results_df = df_with_code.copy()
    results_df['anomaly_score'] = scores
    results_df['is_anomaly'] = (predictions == -1)
    results_df['prediction'] = predictions
    results_df['ground_truth'] = y_true_binary
    results_df['correct'] = (y_true_binary == y_pred_binary)
    
    output_path = 'data/processed/evaluation_results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ Kết quả đã được lưu vào {output_path}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'results_df': results_df
    }


def evaluate_autoencoder():
    """Đánh giá Autoencoder model"""
    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow không được cài đặt")
    
    print("=" * 50)
    print("DANH GIA AUTOENCODER MODEL")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH_AE):
        raise FileNotFoundError(f"Model không tồn tại tại {MODEL_PATH_AE}. Vui lòng train model trước.")
    
    print("Đang load model...")
    autoencoder = keras.models.load_model(MODEL_PATH_AE)
    meta_ae = joblib.load('models/autoencoder_meta.pkl')
    scaler = meta_ae['scaler']
    emb_model_name = meta_ae.get('emb_model_name', EMB_MODEL)
    
    print("Đang load dữ liệu...")
    df = load_data()
    
    # Lọc chỉ lấy các dòng có code (giống như training)
    df_with_code = df[df['code'].notna() & (df['code'].astype(str).str.strip() != '')]
    if len(df_with_code) == 0:
        raise ValueError("Không có dòng nào có code trong dữ liệu!")
    
    print(f"  - Tổng số dòng: {len(df)}")
    print(f"  - Dòng có code: {len(df_with_code)}")
    
    print("Đang tạo embeddings từ CODE (giống như training)...")
    texts = df_with_code['code'].astype(str).tolist()
    embeddings = create_embeddings(texts, use_codebert=use_codebert, model_name=emb_model_name)
    
    # Chuẩn hóa và predict
    emb_scaled = scaler.transform(embeddings)
    emb_pred = autoencoder.predict(emb_scaled, verbose=0)
    
    # Tính reconstruction error
    reconstruction_error = np.mean(np.square(emb_scaled - emb_pred), axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    
    # Predictions: error > threshold = anomaly
    predictions = np.where(reconstruction_error > threshold, -1, 1)
    
    # Tạo ground truth (sử dụng df_with_code thay vì df)
    y_true, labels_binary = create_ground_truth_labels(df_with_code)
    y_pred_binary = np.where(predictions == -1, 1, 0)
    y_true_binary = labels_binary
    
    # Tính metrics
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    print(f"\nReconstruction Error Threshold: {threshold:.4f}")
    print(f"Anomalies phát hiện: {np.sum(predictions == -1)}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'threshold': threshold
    }


def plot_score_distribution(scores, save_path='models/score_distribution.png'):
    """Vẽ biểu đồ phân phối scores"""
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Threshold (0)')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Biểu đồ đã được lưu vào {save_path}")
    plt.close()


if __name__ == '__main__':
    import sys
    
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'if'
    
    if model_type == 'ae':
        evaluate_autoencoder()
    else:
        results = evaluate_isolation_forest()
        plot_score_distribution(results['results_df']['anomaly_score'])

