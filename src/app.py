"""
Streamlit App - GenAI Blockchain Security
Hệ thống phát hiện lỗ hổng Smart Contract sử dụng Supervised Learning & CodeBERT
"""
import os
import sys
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_qa import retrieve, compose_prompt
from src.model_training import create_embeddings, USE_CODEBERT, CODEBERT_MODEL, EMB_MODEL

# Config
st.set_page_config(
    page_title="GenAI Security Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH_CLF = 'models/trained_classifier.pkl'
DATA_PATH = 'data/processed/findings.csv'
META_PATH = 'data/processed/metadf.parquet'

# Styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E88E5;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #424242;}
    .card {padding: 1.5rem; border-radius: 10px; background-color: #f8f9fa; border: 1px solid #e0e0e0; margin-bottom: 1rem;}
    .safe {color: #2e7d32; font-weight: bold;}
    .vuln {color: #c62828; font-weight: bold;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_resources():
    """Load model and resources once"""
    try:
        if not os.path.exists(MODEL_PATH_CLF):
            return None
        return joblib.load(MODEL_PATH_CLF)
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None

# Load resources
model_meta = load_model_resources()


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-shield-green.png", width=64)
    st.title("GenAI Security")
    st.markdown("### Blockchain Security Scanner")
    st.markdown("---")
    
    menu = st.radio(
        "Menu:",
        ["📊 Dashboard", "🔍 Smart Scan", "🤖 AI Assistant (RAG)"],
        index=0
    )
    
    st.markdown("---")
    if model_meta:
        st.success("✓ Model đã sẵn sàng")
        st.caption(f"Model: {model_meta.get('type', 'Unknown')}")
        st.caption(f"Embeddings: {'CodeBERT' if model_meta.get('use_codebert') else 'SentenceTransformer'}")
    else:
        st.warning("⚠️ Model chưa khả dụng")
        st.caption("Hãy chạy training trước")


# --- TAB 1: DASHBOARD ---
if menu == "📊 Dashboard":
    st.markdown('<p class="main-header">📊 Security Overview</p>', unsafe_allow_html=True)
    
    # Load data stats
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            n_vuln = df['impact'].astype(str).str.upper().eq('HIGH').sum()
            st.metric("High Severity", n_vuln, delta_color="inverse")
        with col3:
            n_code = df['code'].notna().sum()
            st.metric("Code Snippets", n_code)
        with col4:
            n_funcs = df['function_name'].nunique()
            st.metric("Unique Functions", n_funcs)
            
        st.markdown("---")
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Phân bố mức độ nghiêm trọng (Impact)")
            if 'impact' in df.columns:
                st.bar_chart(df['impact'].value_counts())
        
        with c2:
            st.subheader("Top Vulnerability Types")
            if 'vulnerability_label' in df.columns:
                top_vulns = df['vulnerability_label'].value_counts().head(5)
                st.write(top_vulns)
                
    else:
        st.info("Chưa có dữ liệu. Vui lòng chạy pipeline xử lý dữ liệu trước.")


# --- TAB 2: SMART SCAN ---
elif menu == "🔍 Smart Scan":
    st.markdown('<p class="main-header">🔍 Smart Contract Scanner</p>', unsafe_allow_html=True)
    st.markdown("Phân tích mã nguồn Solidity sử dụng **AI Supervised Learning**.")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("Source Code Input")
        code_input = st.text_area(
            "Paste Solidity code here:",
            height=400,
            placeholder="contract MyToken {\n    mapping(address => uint) balances;\n    ...\n}"
        )
        
        analyze_btn = st.button("🚀 Analyze Security", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Analysis Results")
        
        if analyze_btn and code_input:
            if not model_meta:
                st.error("Model chưa được load. Vui lòng kiểm tra lại quá trình training.")
            else:
                try:
                    with st.spinner("Đang phân tích vector code..."):
                        # 1. Feature Extraction (Embeddings)
                        clf = model_meta['clf']
                        use_bert = model_meta.get('use_codebert', True)
                        
                        # Create embedding for input
                        emb = create_embeddings([code_input], use_codebert=use_bert)
                        
                        # 2. Prediction
                        prob = clf.predict_proba(emb)[0] # [Prob_Safe, Prob_Vuln]
                        is_vuln = prob[1] > 0.5
                        confidence = prob[1] if is_vuln else prob[0]
                        
                        # 3. Display
                        st.markdown("---")
                        if is_vuln:
                            st.error(f"🚨 PHÁT HIỆN NGUY CƠ BẢO MẬT")
                            st.metric("Mức độ rủi ro", f"{prob[1]*100:.1f}%", delta="High Risk", delta_color="inverse")
                            st.error("Code này có các đặc trưng giống với các lỗ hổng đã biết.")
                        else:
                            st.success(f"✅ AN TOÀN CAO")
                            st.metric("Độ an toàn", f"{prob[0]*100:.1f}%", delta="Safe")
                            st.success("Không tìm thấy mẫu lỗ hổng phổ biến.")
                            
                        # Explanation (Fake LIME for demo or Real feature highlights if imp)
                        with st.expander("Chi tiết kỹ thuật"):
                            st.write(f"- **Algorithm**: RandomForest Classifier")
                            st.write(f"- **Embedding**: {model_meta.get('emb_model_name')}")
                            st.write(f"- **Vector Size**: {emb.shape[1]} dimensions")
                            
                except Exception as e:
                    st.error(f"Lỗi phân tích: {e}")
        else:
            if not code_input:
                st.info("👈 Nhập code để bắt đầu phân tích")


# --- TAB 3: AI ASSISTANT (RAG) ---
elif menu == "🤖 AI Assistant (RAG)":
    st.markdown('<p class="main-header">🤖 Security Assistant</p>', unsafe_allow_html=True)
    st.markdown("Hỏi đáp về các lỗ hổng bảo mật và cách phòng tránh dựa trên cơ sở tri thức (Knowledge Base).")
    
    query = st.chat_input("Hỏi gì đó về bảo mật smart contract...")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query:
        # User message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Bot response
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm thông tin..."):
                # 1. Retrieve docs
                docs = retrieve(query, k=3)
                
                if docs:
                    # 2. Construct prompt (simple version for demo)
                    context_str = "\n\n".join([f"Doc {i+1}: {d['content'][:500]}..." for i, d in enumerate(docs)])
                    
                    response_text = f"**Dựa trên cơ sở dữ liệu của chúng tôi:**\n\n"
                    for i, d in enumerate(docs):
                        response_text += f"- **{d['title']}**: {d.get('impact', 'N/A')}\n"
                    
                    response_text += "\n\n💡 *Gợi ý: Bạn có thể copy context này vào ChatGPT nếu cần câu trả lời chi tiết hơn.*"
                    
                    # Optional: Expanders for full context
                    with st.expander("Xem chi tiết tài liệu tham khảo"):
                        for d in docs:
                            st.info(f"**{d['title']}**\n\n{d['content'][:300]}...")
                            
                else:
                    response_text = "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
