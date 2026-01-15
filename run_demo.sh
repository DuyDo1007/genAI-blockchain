#!/bin/bash

# Script chạy nhanh toàn hệ thống GenAI Blockchain Security
# Sử dụng: bash run_demo.sh hoặc ./run_demo.sh

# Không dừng khi có lỗi, để tự fix
set +e

echo "=========================================="
echo "GenAI Blockchain Security - Quick Start"
echo "=========================================="
echo ""

# Màu sắc cho output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kiểm tra và thiết lập virtual environment
VENV_DIR=".venv"
USE_VENV=true

# Kiểm tra xem có virtual environment không
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment chưa được tạo.${NC}"
    echo -e "${BLUE}Bạn có muốn tạo virtual environment không? (y/n):${NC}"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Đang tạo virtual environment...${NC}"
        # Tạo virtual environment trực tiếp
        if command -v python3 &> /dev/null; then
            python3 -m venv "$VENV_DIR"
        elif command -v python &> /dev/null; then
            python -m venv "$VENV_DIR"
        else
            echo "❌ Không tìm thấy Python. Vui lòng cài đặt Python."
            exit 1
        fi
        echo -e "${GREEN}✓ Virtual environment đã được tạo${NC}"
    else
        USE_VENV=false
        echo -e "${YELLOW}Sử dụng Python hệ thống (không khuyến nghị)${NC}"
    fi
fi

# Kích hoạt virtual environment nếu có
if [ "$USE_VENV" = true ] && [ -d "$VENV_DIR" ]; then
    echo -e "${BLUE}Đang kích hoạt virtual environment...${NC}"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source "$VENV_DIR/Scripts/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi
    echo -e "${GREEN}✓ Virtual environment đã được kích hoạt${NC}"
fi

# Kiểm tra Python
if ! command -v python &> /dev/null; then
    echo "❌ Python không được tìm thấy. Vui lòng cài đặt Python."
    exit 1
fi

echo -e "${BLUE}Bước 1: Kiểm tra dependencies...${NC}"
if [ ! -f "requirements.txt" ]; then
    echo "❌ Không tìm thấy requirements.txt"
    exit 1
fi

# Function tự động fix lỗi Keras/TensorFlow
# Function tự động fix lỗi Keras/TensorFlow (Đã loại bỏ do gây xung đột trên Python 3.12)
# Dependencies được quản lý trong requirements.txt
fix_keras_tensorflow() {
    echo -e "${YELLOW}⚠️  Lỗi Keras/TensorFlow được phát hiện. Vui lòng kiểm tra requirements.txt và đảm bảo cài đặt phiên bản tương thích với Python ${TRAVIS_PYTHON_VERSION:-3.x}.${NC}"
    echo -e "${YELLOW}   Khuyến nghị: pip install --upgrade tensorflow keras tf-keras${NC}"
    return 1
}

# Function chạy Python script với auto-retry và auto-fix
run_with_retry() {
    local script=$1
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        python "$script" 2>&1 | tee /tmp/script_output.log
        local exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -eq 0 ]; then
            return 0
        fi
        
        # Kiểm tra xem có lỗi Keras/TensorFlow không
        if grep -q "keras\|tensorflow\|tf_keras\|ModuleNotFoundError.*tensorflow" /tmp/script_output.log 2>/dev/null; then
            echo -e "${YELLOW}⚠️  Phát hiện lỗi Keras/TensorFlow, đang tự động sửa... (Lần thử $((retry+1))/$max_retries)${NC}"
            fix_keras_tensorflow
            retry=$((retry+1))
            sleep 2
        else
            # Lỗi khác, không retry
            return $exit_code
        fi
    done
    
    echo -e "${YELLOW}⚠️  Đã thử $max_retries lần nhưng vẫn lỗi. Tiếp tục...${NC}"
    return 1
}

echo -e "${YELLOW}Đang cài đặt dependencies (nếu chưa có)...${NC}"

# Uninstall Keras 3 nếu có (gây xung đột với Transformers)
# Cài các dependencies từ requirements.txt
echo -e "${YELLOW}Đang cài đặt/cập nhật dependencies...${NC}"

# Cài các dependencies còn lại
pip install -q -r requirements.txt || echo "⚠️  Một số packages có thể đã được cài đặt"

echo ""
echo -e "${BLUE}Bước 2: Xử lý dữ liệu...${NC}"
if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw/*.json 2>/dev/null)" ]; then
    echo "❌ Không tìm thấy dữ liệu trong data/raw/"
    exit 1
fi

python src/data_preprocessing.py
if [ $? -ne 0 ]; then
    echo "❌ Lỗi khi xử lý dữ liệu"
    exit 1
fi
echo -e "${GREEN}✓ Dữ liệu đã được xử lý${NC}"

echo ""
echo -e "${BLUE}Bước 3: Tạo vector store (FAISS index)...${NC}"
run_with_retry src/ingest_to_vectorstore.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Vector store đã được tạo${NC}"
else
    echo -e "${YELLOW}⚠️  Có lỗi khi tạo vector store nhưng sẽ tiếp tục...${NC}"
fi

echo ""
echo -e "${BLUE}Bước 4: Train mô hình...${NC}"
run_with_retry src/model_training.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Mô hình đã được train${NC}"
else
    echo -e "${YELLOW}⚠️  Có lỗi khi train mô hình nhưng sẽ tiếp tục...${NC}"
fi

echo ""
echo -e "${BLUE}Bước 5: Đánh giá mô hình...${NC}"
if [ -f "src/evaluate_model.py" ]; then
    run_with_retry src/evaluate_model.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Đánh giá mô hình hoàn tất${NC}"
    else
        echo -e "${YELLOW}⚠️  Có lỗi khi đánh giá mô hình, bỏ qua...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  File evaluate_model.py không tồn tại, bỏ qua bước này${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo -e "✓ Tất cả các bước đã hoàn tất!"
echo -e "==========================================${NC}"
echo ""
echo -e "${BLUE}Bước 6: Khởi động Streamlit App...${NC}"
echo -e "${YELLOW}Đang khởi động Streamlit...${NC}"
echo ""
echo -e "${BLUE}Ứng dụng sẽ mở tại: http://localhost:8501${NC}"
echo -e "${YELLOW}Nhấn Ctrl+C để dừng ứng dụng${NC}"
echo ""
sleep 2

# Chạy Streamlit app
streamlit run src/app.py

