import os
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest

META_PATH = 'data/processed/metadf.parquet'
CSV_PATH = 'data/processed/findings.csv'
MODEL_OUT_IF = 'models/trained_if.pkl'
EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def load_data():
    if os.path.exists(META_PATH):
        return pd.read_parquet(META_PATH)
    elif os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError(f"Data not found at {META_PATH} or {CSV_PATH}")


def train_isolation_forest(contamination=0.15, random_state=42):
    print("=" * 50)
    print("Training IsolationForest Model")
    print("=" * 50)
    
    df = load_data()
    print(f"Loaded {len(df)} samples")
    texts = (df['title'].fillna('') + '\n' + df['content'].fillna('')).astype(str).tolist()
    
    print("Creating embeddings...")
    model = SentenceTransformer(EMB_MODEL)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print(f"Training IsolationForest (contamination={contamination})...")
    clf = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    clf.fit(embeddings)
    
    os.makedirs('models', exist_ok=True)
    meta = {
        'clf': clf,
        'emb_model_name': EMB_MODEL,
        'contamination': contamination,
        'emb_shape': embeddings.shape
    }
    joblib.dump(meta, MODEL_OUT_IF)
    print(f"✓ Model saved to {MODEL_OUT_IF}")
    
    predictions = clf.predict(embeddings)
    n_anomalies = np.sum(predictions == -1)
    print(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(predictions)*100:.2f}%)")
    
    return meta


if __name__ == '__main__':
    train_isolation_forest()