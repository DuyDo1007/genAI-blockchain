"""
Huấn luyện mô hình phát hiện bất thường (Anomaly Detection)
Hỗ trợ IsolationForest và Autoencoder
"""
import os
import sys
import io
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Thử import transformers cho CodeBERT
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers không khả dụng, sẽ chỉ dùng SentenceTransformer")

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


META_PATH = 'data/processed/metadf.parquet'
CSV_PATH = 'data/processed/findings.csv'
MODEL_OUT_IF = 'models/trained_if.pkl'
MODEL_OUT_AE = 'models/autoencoder.h5'

# Embedding models - có thể chọn giữa SentenceTransformer và CodeBERT
EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'  # Default
CODEBERT_MODEL = 'microsoft/codebert-base'  # CodeBERT model
USE_CODEBERT = True  # Set True để dùng CodeBERT, False để dùng SentenceTransformer


def load_data():
    """
    Load dữ liệu từ CSV hoặc Parquet
    """
    if os.path.exists(META_PATH):
        df = pd.read_parquet(META_PATH)
    elif os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tại {META_PATH} hoặc {CSV_PATH}")
    
    return df


def create_embeddings_codebert(texts, model_name=CODEBERT_MODEL, batch_size=8, max_length=512):
    """
    Tạo embeddings từ CodeBERT model - Corrected pooling
    
    Args:
        texts: List các text cần embed
        model_name: Tên model CodeBERT
        batch_size: Batch size cho encoding
        max_length: Độ dài tối đa của sequence
    
    Returns:
        numpy array chứa embeddings
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers không khả dụng. Cần cài: pip install transformers torch")
    
    print(f"Đang load CodeBERT model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        raise e
    
    # Set model to evaluation mode
    model.eval()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  - Sử dụng device: {device}")
    
    embeddings = []
    
    try:
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize (trả về cả attention_mask)
            try:
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
            except Exception as e:
                print(f"Lỗi khi tokenize batch {i}: {e}")
                continue
            
            # Move to device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                # Use mixed precision if available/beneficial
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(**encoded)
                    
                    # Correct Mean Pooling using attention_mask
                    # Mask padding tokens so they don't contribute to the average
                    input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    
                    # Sum embeddings of non-padding tokens
                    sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
                    
                    # Count non-padding tokens (prevent division by zero)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Calculate mean
                    batch_embeddings = (sum_embeddings / sum_mask).float().cpu().numpy()
            
            embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Đã xử lý {i + len(batch_texts)}/{len(texts)} samples...")

    except KeyboardInterrupt:
        print("\n⚠️  Quá trình bị dừng bởi người dùng (Ctrl+C).")
        sys.exit(0)
    
    return np.vstack(embeddings)


def create_embeddings(texts, use_codebert=None, model_name=None):
    """
    Tạo embeddings từ text, hỗ trợ cả SentenceTransformer và CodeBERT
    
    Args:
        texts: List các text cần embed
        use_codebert: True để dùng CodeBERT, False để dùng SentenceTransformer, None để dùng default
        model_name: Tên model cụ thể (optional)
    
    Returns:
        numpy array chứa embeddings
    """
    if use_codebert is None:
        use_codebert = USE_CODEBERT
    
    if use_codebert and TRANSFORMERS_AVAILABLE:
        # Sử dụng CodeBERT
        if model_name is None:
            model_name = CODEBERT_MODEL
        print(f"✓ Sử dụng CodeBERT: {model_name}")
        return create_embeddings_codebert(texts, model_name)
    else:
        # Sử dụng SentenceTransformer
        if model_name is None:
            model_name = EMB_MODEL
        print(f"✓ Sử dụng SentenceTransformer: {model_name}")
        model = SentenceTransformer(model_name)
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def train_isolation_forest(contamination=0.05, random_state=42):
    """
    Huấn luyện IsolationForest model
    
    Args:
        contamination: Tỷ lệ anomalies trong dữ liệu (0.05 = 5%)
        random_state: Random seed
    
    Returns:
        Trained model và metadata
    """
    print("=" * 50)
    print("Huấn luyện IsolationForest Model")
    print("=" * 50)
    
    df = load_data()
    print(f"Đang load {len(df)} samples...")
    
    # Lọc chỉ lấy các dòng có code
    df_with_code = df[df['code'].notna() & (df['code'].astype(str).str.strip() != '')]
    if len(df_with_code) == 0:
        raise ValueError("Không có dòng nào có code trong dữ liệu! Vui lòng chạy lại data_preprocessing.py để tạo CSV với code.")
    
    print(f"  - Tổng số dòng: {len(df)}")
    print(f"  - Dòng có code: {len(df_with_code)}")
    
    # Tạo embeddings từ cột CODE
    texts = df_with_code['code'].astype(str).tolist()
    
    print("Đang tạo embeddings từ code...")
    emb = create_embeddings(texts, use_codebert=USE_CODEBERT)
    print(f"Embeddings shape: {emb.shape}")
    
    # Train IsolationForest
    print(f"Đang train IsolationForest với contamination={contamination}...")
    clf = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    clf.fit(emb)
    
    # Lưu model
    os.makedirs('models', exist_ok=True)
    emb_model_used = CODEBERT_MODEL if USE_CODEBERT else EMB_MODEL
    meta = {
        'clf': clf,
        'emb_model_name': emb_model_used,
        'use_codebert': USE_CODEBERT,
        'contamination': contamination,
        'emb_shape': emb.shape
    }
    joblib.dump(meta, MODEL_OUT_IF)
    print(f"✓ Đã lưu IsolationForest model vào {MODEL_OUT_IF}")
    
    # Thống kê
    predictions = clf.predict(emb)
    n_anomalies = np.sum(predictions == -1)
    print(f"  - Anomalies phát hiện: {n_anomalies} ({n_anomalies/len(predictions)*100:.2f}%)")
    
    return meta


def train_autoencoder(encoding_dim=128, epochs=200, batch_size=32, validation_split=0.2):
    """
    Huấn luyện Autoencoder model
    
    Args:
        encoding_dim: Kích thước encoding layer
        epochs: Số epochs
        batch_size: Batch size
        validation_split: Tỷ lệ validation split
    
    Returns:
        Trained model
    """
    try:
        import tensorflow as tf
        # Sử dụng tf.keras thay vì import riêng để tránh lỗi tương thích
        keras = tf.keras
        layers = tf.keras.layers
    except ImportError:
        raise ImportError("TensorFlow không được cài đặt. Cần cài: pip install tensorflow")
    
    print("=" * 50)
    print("Huấn luyện Autoencoder Model")
    print("=" * 50)
    
    df = load_data()
    print(f"Đang load {len(df)} samples...")
    
    # Lọc chỉ lấy các dòng có code
    df_with_code = df[df['code'].notna() & (df['code'].astype(str).str.strip() != '')]
    if len(df_with_code) == 0:
        raise ValueError("Không có dòng nào có code trong dữ liệu! Vui lòng chạy lại data_preprocessing.py để tạo CSV với code.")
    
    print(f"  - Tổng số dòng: {len(df)}")
    print(f"  - Dòng có code: {len(df_with_code)}")
    
    # Tạo embeddings từ cột CODE
    texts = df_with_code['code'].astype(str).tolist()
    
    print("Đang tạo embeddings từ code...")
    emb = create_embeddings(texts, use_codebert=USE_CODEBERT)
    print(f"Embeddings shape: {emb.shape}")
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)
    
    # Split train/validation
    X_train, X_val = train_test_split(emb_scaled, test_size=validation_split, random_state=42)
    
    input_dim = emb_scaled.shape[1]
    
    # Xây dựng Autoencoder với kiến trúc cải thiện
    print(f"Đang xây dựng Autoencoder (input_dim={input_dim}, encoding_dim={encoding_dim})...")
    print("  - Kiến trúc: Input -> 256 -> 128 -> encoding_dim -> 128 -> 256 -> Output")
    print("  - Sử dụng Dropout và BatchNormalization để tránh overfitting")
    
    # Sử dụng Sequential API để tránh lỗi tương thích
    # Encoder layers
    encoder_layers = [
        layers.Dense(256, activation='relu', input_shape=(input_dim,), name='encoder_1'),
        layers.BatchNormalization(name='encoder_bn_1'),
        layers.Dropout(0.3, name='encoder_dropout_1'),
        layers.Dense(128, activation='relu', name='encoder_2'),
        layers.BatchNormalization(name='encoder_bn_2'),
        layers.Dropout(0.2, name='encoder_dropout_2'),
        layers.Dense(encoding_dim, activation='relu', name='encoded')
    ]
    
    # Decoder layers
    decoder_layers = [
        layers.Dense(128, activation='relu', name='decoder_1'),
        layers.BatchNormalization(name='decoder_bn_1'),
        layers.Dropout(0.2, name='decoder_dropout_1'),
        layers.Dense(256, activation='relu', name='decoder_2'),
        layers.BatchNormalization(name='decoder_bn_2'),
        layers.Dropout(0.3, name='decoder_dropout_2'),
        layers.Dense(input_dim, activation='sigmoid', name='decoded')
    ]
    
    # Tạo autoencoder (encoder + decoder)
    autoencoder = keras.Sequential(encoder_layers + decoder_layers, name='autoencoder')
    
    # Tạo encoder riêng (chỉ lấy phần encoder)
    encoder = keras.Sequential(encoder_layers, name='encoder')
    
    # Compile với learning rate thấp hơn và L2 regularization
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    # Train
    print(f"Đang train Autoencoder ({epochs} epochs)...")
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Lưu model
    os.makedirs('models', exist_ok=True)
    autoencoder.save(MODEL_OUT_AE)
    
    # Tạo encoder từ autoencoder đã train
    # Lấy các layers encoder từ autoencoder (7 layers đầu tiên)
    encoder_layers_trained = []
    for i in range(len(encoder_layers)):
        layer = autoencoder.layers[i]
        encoder_layers_trained.append(layer)
    
    # Tạo encoder mới với các layers đã train
    encoder = keras.Sequential(encoder_layers_trained, name='encoder')
    
    # Lưu metadata
    emb_model_used = CODEBERT_MODEL if USE_CODEBERT else EMB_MODEL
    meta_ae = {
        'model_path': MODEL_OUT_AE,
        'emb_model_name': emb_model_used,
        'use_codebert': USE_CODEBERT,
        'encoding_dim': encoding_dim,
        'input_dim': input_dim,
        'scaler': scaler,
        'encoder': encoder
    }
    joblib.dump(meta_ae, 'models/autoencoder_meta.pkl')
    
    print(f"✓ Đã lưu Autoencoder model vào {MODEL_OUT_AE}")
    
    # Tính reconstruction error trên validation set
    X_val_pred = autoencoder.predict(X_val)
    reconstruction_error = np.mean(np.square(X_val - X_val_pred), axis=1)
    threshold = np.percentile(reconstruction_error, 95)  # 95th percentile làm threshold
    
    print(f"  - Reconstruction error threshold: {threshold:.4f}")
    print(f"  - Anomalies (error > threshold): {np.sum(reconstruction_error > threshold)}")
    
    return autoencoder, meta_ae


def train_models(model_type='both', **kwargs):
    """
    Huấn luyện mô hình phát hiện bất thường
    
    Args:
        model_type: 'isolation_forest', 'autoencoder', hoặc 'both'
        **kwargs: Các tham số cho từng model
    """
    if model_type in ['isolation_forest', 'both']:
        train_isolation_forest(
            contamination=kwargs.get('contamination', 0.05),
            random_state=kwargs.get('random_state', 42)
        )
    
    if model_type in ['autoencoder', 'both']:
        train_autoencoder(
            encoding_dim=kwargs.get('encoding_dim', 128),
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 32),
            validation_split=kwargs.get('validation_split', 0.2)
        )
    
    if model_type in ['classifier', 'all']:
        train_classifier(
            n_estimators=kwargs.get('n_estimators', 100)
        )


def extract_code_features(df):
    """
    Trích xuất features từ code (Utility shared with evaluate_model)
    """
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
    features['has_reentrancy'] = code_col.str.contains(r'reentrancy', case=False, na=False)
    
    return features


def create_ground_truth_labels(df):
    """
    Tạo labels cho converting to supervised problem
    """
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
            
    # 2. Code features based
    if 'code' in df.columns:
        code_features = extract_code_features(df)
        if 'code_length' in code_features:
            long_code = code_features['code_length'] > code_features['code_length'].quantile(0.75)
            has_vuln_patterns = (
                code_features.get('has_transfer', pd.Series([False]*len(df))) |
                code_features.get('has_call', pd.Series([False]*len(df))) |
                code_features.get('has_delegatecall', pd.Series([False]*len(df)))
            )
            labels[long_code & has_vuln_patterns] = 1
            
    return labels


def train_classifier(n_estimators=100, random_state=42):
    """
    Huấn luyện Supervised Classifier (RandomForest)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    print("=" * 50)
    print("Huấn luyện Classifier (Supervised)")
    print("=" * 50)
    
    df = load_data()
    df_with_code = df[df['code'].notna() & (df['code'].astype(str).str.strip() != '')]
    if len(df_with_code) == 0:
        raise ValueError("No code found")
        
    print(f"  - Dữ liệu: {len(df_with_code)} samples")
    
    # Prepare Features (Embeddings)
    texts = df_with_code['code'].astype(str).tolist()
    print("Đang tạo embeddings...")
    emb = create_embeddings(texts, use_codebert=USE_CODEBERT)
    
    # Prepare Labels
    y = create_ground_truth_labels(df_with_code)
    print(f"  - Labels distribution: Safe={np.sum(y==0)}, Vuln={np.sum(y==1)}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(emb, y, test_size=0.2, random_state=random_state)
    
    # Train
    print(f"Đang train RandomForest ({n_estimators} estimators)...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate briefly
    y_pred = clf.predict(X_test)
    print("Kết quả trên tập test split:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Vuln']))
    
    # Save
    os.makedirs('models', exist_ok=True)
    emb_model_used = CODEBERT_MODEL if USE_CODEBERT else EMB_MODEL
    meta = {
        'clf': clf,
        'emb_model_name': emb_model_used,
        'use_codebert': USE_CODEBERT,
        'type': 'classifier'
    }
    joblib.dump(meta, 'models/trained_classifier.pkl')
    print(f"✓ Đã lưu Classifier vào models/trained_classifier.pkl")
    return meta


if __name__ == '__main__':
    # Mặc định train tất cả
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    if model_type == 'if':
        train_isolation_forest()
    elif model_type == 'ae':
        train_autoencoder()
    elif model_type == 'classifier':
        train_classifier()
    else:
        print("Huan luyen TOAN BO mo hinh (Autoencoder, IsolationForest, Classifier)...")
        train_models(model_type='both') # call existing
        train_classifier() # call new