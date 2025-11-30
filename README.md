# GenAI for Blockchain Security

Hệ thống phân tích và phát hiện lỗ hổng bảo mật trong Smart Contracts sử dụng AI.

**Stack**: Sentence-Transformers + FAISS + IsolationForest + Streamlit + OpenAI API

## 🎯 Tính năng

### 1. **RAG Q&A** - Hỏi đáp về Smart Contract Security

- Truy vấn kiến thức từ 912 audit findings
- Sử dụng FAISS vector store để tìm documents liên quan
- Generate câu trả lời bằng OpenAI API
- User chỉ cần nhập API key

### 2. **Anomaly Detection** - Phát hiện bất thường

- Phát hiện findings bất thường trong smart contracts
- Sử dụng IsolationForest model (15% contamination)
- Anomaly Score: < 0 = anomaly, ≥ 0 = normal
- Hỗ trợ batch processing

### 3. **Data Processing** - Xử lý dữ liệu

- Chuyển đổi 912 JSON files → CSV
- Trích xuất contract name, function name
- Tạo embeddings (384-dimensional)

## 📁 Cấu trúc

```
genai-blockchain-security/
├── data/
│   ├── raw/                           # 912 JSON files
│   └── processed/
│       ├── findings.csv               # Processed data
│       ├── faiss_index.bin            # Vector store
│       ├── metadf.parquet             # Metadata
│       ├── evaluation_results.csv     # Model results
│       └── score_distribution.png     # Anomaly scores chart
├── models/
│   ├── trained_if.pkl                 # IsolationForest model
│   └── score_distribution.png         # Evaluation plot
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── app.py                         # Streamlit UI (3 tabs)
│   ├── data_preprocessing.py          # JSON → CSV conversion
│   ├── ingest_to_vectorstore.py       # Create FAISS index
│   ├── model_training.py              # Train IsolationForest
│   ├── evaluate_model.py              # Model evaluation
│   ├── rag_qa.py                      # RAG functions
│   └── __init__.py
├── requirements.txt
├── run_demo.sh                        # Auto run all steps
└── README.md
```

## 🚀 Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Linux/Mac:
source .venv/bin/activate
# Windows (Git Bash):
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
bash run_demo.sh
```

This will automatically:

- ✓ Preprocess data (JSON → CSV)
- ✓ Create FAISS vector store
- ✓ Train IsolationForest model
- ✓ Evaluate model
- ✓ Launch Streamlit app at http://localhost:8501

### 3. Or Run Individual Steps

```bash
# Step 1: Data preprocessing
python src/data_preprocessing.py

# Step 2: Create vector store
python src/ingest_to_vectorstore.py

# Step 3: Train model
python src/model_training.py

# Step 4: Evaluate model
python src/evaluate_model.py

# Step 5: Launch app
streamlit run src/app.py
```

## 📊 Model Performance

**IsolationForest (contamination=0.15):**

- Anomalies Detected: 137/912 (15.02%)
- Precision: 0.0876
- Recall: 0.0828
- F1 Score: 0.0851
- Accuracy: 0.7171

**Anomaly Score Distribution:**

- Range: -0.0291 to 0.0548
- Mean: 0.0145
- Std: 0.0141
- Threshold: 0 (< 0 = anomaly)

## 🎨 Streamlit UI

### Tab 1: Upload Contract

- Upload JSON/CSV files
- Display contract info
- View statistics

### Tab 2: RAG Q&A (Full RAG)

- Enter OpenAI API key
- Ask questions about smart contract security
- Retrieve 1-10 documents
- Auto-generate answers
- View retrieved documents

### Tab 3: Anomaly Detection

- Single prediction: Paste finding text
- Batch prediction: Upload CSV file
- Get anomaly score and classification

## 💻 Usage Examples

### RAG Q&A

```python
from src.rag_qa import rag_query

result = rag_query(
    query="What is reentrancy vulnerability?",
    api_key="sk-...",
    k=5
)

print(result['answer'])
print(result['documents'])
```

### Anomaly Detection

```python
import joblib
from sentence_transformers import SentenceTransformer

# Load model
meta = joblib.load('models/trained_if.pkl')
clf = meta['clf']
model = SentenceTransformer(meta['emb_model_name'])

# Predict
text = "Your finding text"
emb = model.encode([text], convert_to_numpy=True)
score = clf.decision_function(emb)[0]
is_anomaly = clf.predict(emb)[0] == -1

print(f"Score: {score:.4f}")
print(f"Anomaly: {is_anomaly}")
```

## 📦 Technologies

| Component         | Library               | Version |
| ----------------- | --------------------- | ------- |
| Embeddings        | Sentence-Transformers | 2.7+    |
| Vector Store      | FAISS                 | 1.8+    |
| Anomaly Detection | scikit-learn          | 1.5+    |
| Web UI            | Streamlit             | 1.31+   |
| LLM Integration   | OpenAI                | 1.3+    |
| Data Processing   | pandas, numpy         | Latest  |

## ⚙️ Configuration

### Embedding Model

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Speed: ~0.1ms per text

### IsolationForest Parameters

- contamination: 0.15 (15% expected anomalies)
- n_estimators: 100
- n_jobs: -1 (use all cores)

### FAISS Index

- Type: IndexFlatL2
- Distance metric: L2 (Euclidean)
- Search: O(n\*d) complexity

## 📝 Data Format

**findings.csv** structure:

```
id          | title                          | content           | impact | ...
62000       | Reentrancy Vulnerability       | Description...    | HIGH   | ...
62001       | Integer Overflow in Transfer   | Description...    | MEDIUM | ...
```

## 🔧 Troubleshooting

**Issue**: Scores too close to 0

- **Solution**: Increase contamination in model_training.py (0.15 → 0.2)

**Issue**: FAISS index not found

- **Solution**: Run `python src/ingest_to_vectorstore.py`

**Issue**: Model not trained

- **Solution**: Run `python src/model_training.py`

**Issue**: OpenAI API error

- **Solution**: Check API key, ensure it's valid and has quota

## 📈 Performance

- **Training Time**: ~2 minutes
- **Prediction Time**: <100ms per text
- **Memory Usage**: ~500MB
- **Data Size**: 912 findings × 384 dimensions

## 📄 Files

| File                           | Purpose                             |
| ------------------------------ | ----------------------------------- |
| `src/app.py`                   | Streamlit UI application            |
| `src/model_training.py`        | Train IsolationForest               |
| `src/evaluate_model.py`        | Evaluate model performance          |
| `src/rag_qa.py`                | RAG functions (retrieve + generate) |
| `src/ingest_to_vectorstore.py` | Create FAISS index                  |
| `src/data_preprocessing.py`    | Convert JSON to CSV                 |
| `requirements.txt`             | Python dependencies                 |
| `run_demo.sh`                  | Auto-run all steps                  |

## 🎓 Next Steps

1. **Improve Model**

   - Experiment with different contamination values
   - Try other anomaly detection algorithms
   - Add feature engineering

2. **Enhance RAG**

   - Implement prompt caching
   - Add response quality metrics
   - Fine-tune retrieval threshold

3. **Scale**
   - Use GPU acceleration for embeddings
   - Implement batch processing
   - Add API layer
