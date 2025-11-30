import os
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, accuracy_score
)
import matplotlib.pyplot as plt

MODEL_PATH_IF = 'models/trained_if.pkl'
META_PATH = 'data/processed/metadf.parquet'
CSV_PATH = 'data/processed/findings.csv'
EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def load_data():
    if os.path.exists(META_PATH):
        return pd.read_parquet(META_PATH)
    elif os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError(f"Data not found at {META_PATH} or {CSV_PATH}")


def create_ground_truth_labels(df):
    labels = np.zeros(len(df))
    high_impact_mask = df['impact'].str.upper() == 'HIGH'
    labels[high_impact_mask] = 1
    
    if 'vulnerability_label' in df.columns:
        critical_vulns = ['REENTRANCY', 'OVERFLOW', 'ACCESS_CONTROL', 'UNCHECKED_CALL']
        for vuln in critical_vulns:
            vuln_mask = df['vulnerability_label'].str.upper().str.contains(vuln, na=False)
            labels[vuln_mask] = 1
    
    y_true = np.where(labels == 1, -1, 1)
    return y_true, labels


def evaluate_isolation_forest():
    print("=" * 50)
    print("EVALUATING ISOLATION FOREST MODEL")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH_IF):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH_IF}")
    
    print("Loading model...")
    meta = joblib.load(MODEL_PATH_IF)
    clf = meta['clf']
    emb_model_name = meta.get('emb_model_name', EMB_MODEL)
    
    print("Loading data...")
    df = load_data()
    
    print("Creating embeddings...")
    model = SentenceTransformer(emb_model_name)
    texts = (df['title'].fillna('') + '\n' + df['content'].fillna('')).astype(str).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    print("Making predictions...")
    predictions = clf.predict(embeddings)
    scores = clf.decision_function(embeddings)
    
    y_true, labels_binary = create_ground_truth_labels(df)
    y_pred_binary = np.where(predictions == -1, 1, 0)
    y_true_binary = labels_binary
    
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total samples: {len(predictions)}")
    print(f"Anomalies detected: {np.sum(predictions == -1)} ({np.sum(predictions == -1)/len(predictions)*100:.2f}%)")
    print(f"Normal samples: {np.sum(predictions == 1)} ({np.sum(predictions == 1)/len(predictions)*100:.2f}%)")
    print(f"\nGround Truth:")
    print(f"  Actual anomalies: {np.sum(y_true_binary == 1)} ({np.sum(y_true_binary == 1)/len(y_true_binary)*100:.2f}%)")
    print(f"  Actual normal: {np.sum(y_true_binary == 0)} ({np.sum(y_true_binary == 0)/len(y_true_binary)*100:.2f}%)")
    
    print("\n" + "-" * 50)
    print("METRICS")
    print("-" * 50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    print("\n" + "-" * 50)
    print("CONFUSION MATRIX")
    print("-" * 50)
    print("              Predicted")
    print("            Normal  Anomaly")
    print(f"Actual Normal {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"       Anomaly {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(y_true_binary, y_pred_binary, 
                                target_names=['Normal', 'Anomaly'], zero_division=0))
    
    print("\n" + "-" * 50)
    print("SCORE STATISTICS")
    print("-" * 50)
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std:  {scores.std():.4f}")
    print(f"Min:  {scores.min():.4f}")
    print(f"Max:  {scores.max():.4f}")
    
    print("\n" + "=" * 50)
    print("TOP 10 ANOMALIES")
    print("=" * 50)
    top_anomalies_idx = np.argsort(scores)[:10]
    for i, idx in enumerate(top_anomalies_idx, 1):
        row = df.iloc[idx]
        is_actual = "✓" if y_true_binary[idx] == 1 else "✗"
        print(f"{i}. ID: {row['id']}, Score: {scores[idx]:.4f} {is_actual}")
        print(f"   Title: {row['title'][:80]}...")
        print(f"   Impact: {row.get('impact', 'N/A')}\n")
    
    results_df = df.copy()
    results_df['anomaly_score'] = scores
    results_df['is_anomaly'] = (predictions == -1)
    results_df['prediction'] = predictions
    results_df['ground_truth'] = y_true_binary
    results_df['correct'] = (y_true_binary == y_pred_binary)
    
    output_path = 'data/processed/evaluation_results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Results saved to {output_path}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 
            'accuracy': accuracy, 'cm': cm, 'results_df': results_df}


def plot_score_distribution(scores, save_path='models/score_distribution.png'):
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
    print(f"✓ Plot saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    results = evaluate_isolation_forest()
    plot_score_distribution(results['results_df']['anomaly_score'])

