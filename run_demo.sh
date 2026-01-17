#!/bin/bash

# Script chạy nhanh hệ thống GenAI Blockchain Security (CodeBERT)
# Sử dụng: bash run_demo.sh hoặc ./run_demo.sh

set +e

echo "=========================================="
echo "GenAI Blockchain Security (CodeBERT)"
echo "=========================================="
echo ""

# Màu sắc
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Kiểm tra Virtual Environment
VENV_DIR=".venv"
USE_VENV=true

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment không tồn tại. Đang tạo mới...${NC}"
    python -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Đã tạo virtual environment${NC}"
fi

# Activate Env
if [ -d "$VENV_DIR" ]; then
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source "$VENV_DIR/Scripts/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi
fi

# Dependencies
echo -e "${BLUE}1. Kiểm tra dependencies...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies ready.${NC}"

# Tạo Vector Store
echo ""
echo -e "${BLUE}2. Tạo Vector Store (CodeBERT)...${NC}"
python src/ingest_to_vectorstore.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Lỗi tạo vector store. Vui lòng kiểm tra lại.${NC}"
fi

# Train Model
echo ""
echo -e "${BLUE}3. Train Model Phân Loại (CodeBERT)...${NC}"
python src/model_training.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Lỗi train model. Vui lòng kiểm tra lại.${NC}"
fi

# Evaluatation (Optional - Commented for speed as requested)
# echo ""
# echo -e "${BLUE}4. Đánh giá nhanh model...${NC}"
# python src/evaluate_model.py

# Run App
echo ""
echo -e "${GREEN}=========================================="
echo -e "✓ Ready to launch!"
echo -e "==========================================${NC}"
echo ""
echo -e "${BLUE}Đang mở Streamlit App...${NC}"
streamlit run src/app.py
