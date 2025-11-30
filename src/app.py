import os
import sys
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_qa import retrieve, compose_prompt

MODEL_META_IF = 'models/trained_if.pkl'
EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

st.set_page_config(
    page_title="GenAI Blockchain Security",
    page_icon="🔒",
    layout="wide"
)

st.sidebar.title("🔒 GenAI Blockchain Security")
st.sidebar.markdown("---")
st.sidebar.markdown("### Hệ thống phân tích và phát hiện lỗ hổng bảo mật trong Smart Contracts")
st.sidebar.markdown("---")
st.sidebar.markdown("**Tính năng:**")
st.sidebar.markdown("- 📤 Upload và phân tích Smart Contract")
st.sidebar.markdown("- 💬 RAG Q&A về bảo mật")
st.sidebar.markdown("- 🔍 Phát hiện bất thường (Anomaly Detection)")

st.title("🔒 GenAI for Blockchain Security")
st.markdown("---")


tab1, tab2, tab3 = st.tabs(["📤 Upload Contract", "💬 RAG Q&A", "🔍 Anomaly Detection"])

with tab1:
    st.header("📤 Upload Smart Contract")
    st.markdown("Upload file JSON hoặc CSV chứa smart contract để phân tích")
    
    uploaded_file = st.file_uploader(
        "Chọn file JSON hoặc CSV",
        type=['json', 'csv'],
        help="Upload file chứa smart contract data"
    )
    
    if uploaded_file is not None:
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext == 'json':
                data = json.load(uploaded_file)
                st.success("✓ Đã tải file JSON thành công")
                if isinstance(data, dict):
                    st.subheader("Thông tin Contract:")
                    st.json(data)
                    if 'title' in data:
                        st.info(f"**Title:** {data['title']}")
                    if 'content' in data:
                        st.text_area("Content:", data['content'], height=200)
                    if 'impact' in data:
                        st.warning(f"**Impact:** {data['impact']}")
                elif isinstance(data, list):
                    st.success(f"✓ Đã tải {len(data)} records")
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    
            elif file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
                st.success(f"✓ Đã tải file CSV thành công ({len(df)} rows)")
                st.dataframe(df.head(20))
                st.subheader("Thống kê:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tổng số records", len(df))
                with col2:
                    if 'impact' in df.columns:
                        st.metric("HIGH impact", df['impact'].str.upper().eq('HIGH').sum())
                with col3:
                    if 'vulnerability_label' in df.columns:
                        st.metric("Vulnerabilities", df['vulnerability_label'].notna().sum())
            
            if st.button("🔍 Phân tích Contract", type="primary"):
                st.info("Đang phân tích... (Tính năng này có thể được mở rộng để tích hợp với RAG và Anomaly Detection)")
                
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file: {str(e)}")
    else:
        st.info("👆 Vui lòng upload file JSON hoặc CSV để bắt đầu")


with tab2:
    st.header("💬 RAG Q&A - Hỏi đáp về Smart Contract Security")
    st.markdown("Nhập câu hỏi về bảo mật smart contract, hệ thống sẽ tìm kiếm và trả lời dựa trên tài liệu")
    
    index_path = 'data/processed/faiss_index.bin'
    meta_path = 'data/processed/metadf.parquet'
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        st.warning("⚠️ Vector store chưa được tạo. Vui lòng chạy `python src/ingest_to_vectorstore.py` trước.")
    else:
        openai_key = st.text_input("OpenAI API Key:", type="password", help="Required for generating answers")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Nhập câu hỏi:",
                placeholder="VD: What is reentrancy vulnerability? How to prevent it?",
                key="rag_query"
            )
        with col2:
            k = st.number_input("Số documents (k):", min_value=1, max_value=10, value=3, step=1)
        
        st.markdown("**Câu hỏi mẫu:**")
        sample_queries = [
            "What is reentrancy vulnerability?",
            "How to prevent integer overflow in smart contracts?",
            "What are common access control issues?",
            "Explain unchecked external calls vulnerability"
        ]
        cols = st.columns(len(sample_queries))
        for i, sample_q in enumerate(sample_queries):
            with cols[i]:
                if st.button(f"📌 {sample_q[:30]}...", key=f"sample_{i}"):
                    query = sample_q
                    st.rerun()
        
        if st.button("🔍 Truy vấn RAG", type="primary", key="rag_search") and query and openai_key:
            try:
                from src.rag_qa import rag_query
                
                with st.spinner("Đang truy vấn và tạo câu trả lời..."):
                    result = rag_query(query, api_key=openai_key, k=k)
                
                st.success("✓ Tìm thấy câu trả lời!")
                
                st.subheader("📚 Documents Retrieved:")
                for i, doc in enumerate(result['documents'], 1):
                    with st.expander(f"Document {i}: {doc['title']} (ID: {doc['id']})"):
                        st.text(doc['content'][:1000] + ('...' if len(doc['content']) > 1000 else ''))
                
                st.subheader("🤖 AI Answer:")
                st.markdown(result['answer'])
                    
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.info("Đảm bảo đã chạy `python src/ingest_to_vectorstore.py` để tạo vector store.")
        elif st.button("⚠️ Kiểm tra lỗi", type="secondary", key="rag_warning"):
            if not query:
                st.warning("⚠️ Vui lòng nhập câu hỏi")
            if not openai_key:
                st.warning("⚠️ Vui lòng nhập OpenAI API Key")


with tab3:
    st.header("🔍 Anomaly Detection - Phát hiện bất thường")
    st.markdown("Phát hiện các smart contract findings bất thường hoặc đáng nghi")
    
    st.info("📊 Mô hình: IsolationForest - Phát hiện anomalies dựa trên isolation trees")
    
    text_input = st.text_area(
        "Nhập finding hoặc smart contract snippet:",
        placeholder="Paste finding text hoặc code snippet để kiểm tra...",
        height=200
    )
    
    if st.button("🔍 Phát hiện bất thường", type="primary") and text_input:
        try:
            if not os.path.exists(MODEL_META_IF):
                st.warning(f"⚠️ Model chưa được train. Vui lòng chạy `python src/model_training.py` trước.")
            else:
                with st.spinner("Đang phân tích..."):
                    meta = joblib.load(MODEL_META_IF)
                    clf = meta['clf']
                    emb_model_name = meta.get('emb_model_name', EMB_MODEL)
                    model = SentenceTransformer(emb_model_name)
                    
                    text_emb = model.encode([text_input], convert_to_numpy=True)
                    score = clf.decision_function(text_emb)[0]
                    prediction = clf.predict(text_emb)[0]
                    is_anomaly = prediction == -1
                    
                    st.subheader("📊 Kết quả:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Anomaly Score", f"{score:.4f}")
                    with col2:
                        if is_anomaly:
                            st.metric("Kết quả", "⚠️ BẤT THƯỜNG", delta="Anomaly")
                        else:
                            st.metric("Kết quả", "✓ BÌNH THƯỜNG", delta="Normal")
                    
                    if is_anomaly:
                        st.error("⚠️ **Phát hiện bất thường!** Finding này có thể chứa lỗ hổng bảo mật nghiêm trọng.")
                    else:
                        st.success("✓ **Bình thường** - Finding này không có dấu hiệu bất thường.")
                    
                    st.info(f"""
                    **Giải thích:**
                    - **Anomaly Score:** {score:.4f}
                    - Score < 0: Bất thường (Anomaly)
                    - Score ≥ 0: Bình thường (Normal)
                    - Score càng âm, mức độ bất thường càng cao
                    """)
                    
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
            st.info("Đảm bảo đã train model trước khi sử dụng.")
    
    st.markdown("---")
    st.subheader("📤 Batch Upload")
    batch_file = st.file_uploader("Upload file CSV chứa nhiều findings:", type=['csv'])
    
    if batch_file is not None:
        try:
            df = pd.read_csv(batch_file)
            st.success(f"✓ Đã tải {len(df)} findings")
            
            if st.button("🔍 Phân tích tất cả", type="primary"):
                if not os.path.exists(MODEL_META_IF):
                    st.warning("Model chưa được train.")
                else:
                    meta = joblib.load(MODEL_META_IF)
                    clf = meta['clf']
                    emb_model_name = meta.get('emb_model_name', EMB_MODEL)
                    model = SentenceTransformer(emb_model_name)
                    
                    texts = df['content'].fillna('').astype(str).tolist() if 'content' in df.columns else df.iloc[:, 0].astype(str).tolist()
                    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
                    scores = clf.decision_function(embeddings)
                    predictions = clf.predict(embeddings)
                    
                    df['anomaly_score'] = scores
                    df['is_anomaly'] = (predictions == -1)
                    
                    st.dataframe(df[['anomaly_score', 'is_anomaly']].head(20))
                    st.success(f"Phát hiện {df['is_anomaly'].sum()} anomalies trong {len(df)} findings")
                    
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")

st.markdown("---")
st.markdown("**GenAI for Blockchain Security** - Hệ thống phân tích và phát hiện lỗ hổng bảo mật trong Smart Contracts")