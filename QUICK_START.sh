#!/bin/bash

# ✅ QUICK START - GenAI Blockchain Security
# Chỉ cần chạy các lệnh này

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🚀 GenAI Blockchain Security - Quick Start            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Kiểm tra Python
echo "1️⃣  Kiểm tra Python..."
python --version
echo ""

# 2. Tạo virtual environment
if [ ! -d ".venv" ]; then
    echo "2️⃣  Tạo virtual environment..."
    python -m venv .venv
else
    echo "2️⃣  Virtual environment đã tồn tại"
fi
echo ""

# 3. Kích hoạt
echo "3️⃣  Kích hoạt virtual environment..."
source .venv/Scripts/activate
echo "   ✓ Activated: $VIRTUAL_ENV"
echo ""

# 4. Cài đặt packages
echo "4️⃣  Cài đặt packages..."
pip install -q -r requirements.txt
echo "   ✓ Packages installed"
echo ""

# 5. Kiểm tra syntax
echo "5️⃣  Kiểm tra syntax Python..."
python -m py_compile src/*.py
echo "   ✓ All files OK"
echo ""

# 6. Chạy pipeline
echo "6️⃣  Chạy pipeline..."
echo ""

echo "   → Bước 1: Xử lý data (JSON → CSV)"
python src/data_preprocessing.py
echo ""

echo "   → Bước 2: Tạo vector store (FAISS)"
python src/ingest_to_vectorstore.py
echo ""

echo "   → Bước 3: Train model (IsolationForest)"
python src/model_training.py if
echo ""

echo "   → Bước 4: Đánh giá model"
python src/evaluate_model.py
echo ""

# 7. Khởi động app
echo "7️⃣  Khởi động Streamlit app..."
echo ""
echo "   🌐 Truy cập: http://localhost:8501"
echo "   📱 Nhấn Ctrl+C để dừng"
echo ""
streamlit run src/app.py

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Hoàn thành!                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
