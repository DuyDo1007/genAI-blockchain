#!/bin/bash

# Script run the entire GenAI Blockchain Security system
# Usage: bash run_demo.sh or ./run_demo.sh

set +e

echo "=========================================="
echo "GenAI Blockchain Security - Quick Start"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

VENV_DIR=".venv"
USE_VENV=true

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found.${NC}"
    echo -e "${BLUE}Create virtual environment? (y/n):${NC}"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Creating virtual environment...${NC}"
        if command -v python3 &> /dev/null; then
            python3 -m venv "$VENV_DIR"
        elif command -v python &> /dev/null; then
            python -m venv "$VENV_DIR"
        else
            echo "❌ Python not found. Please install Python."
            exit 1
        fi
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        USE_VENV=false
        echo -e "${YELLOW}Using system Python (not recommended)${NC}"
    fi
fi

if [ "$USE_VENV" = true ] && [ -d "$VENV_DIR" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source "$VENV_DIR/Scripts/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

if ! command -v python &> /dev/null; then
    echo "❌ Python not found."
    exit 1
fi

echo ""
echo -e "${BLUE}Step 1: Installing dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip > /dev/null 2>&1
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found${NC}"
fi

echo ""
echo -e "${BLUE}Step 2: Data preprocessing...${NC}"
if [ -f "src/data_preprocessing.py" ]; then
    python src/data_preprocessing.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Data preprocessing complete${NC}"
    else
        echo -e "${YELLOW}⚠️  Error in preprocessing, skipping...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  File not found, skipping${NC}"
fi

echo ""
echo -e "${BLUE}Step 3: Creating FAISS vector store...${NC}"
if [ -f "src/ingest_to_vectorstore.py" ]; then
    python src/ingest_to_vectorstore.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Vector store created${NC}"
    else
        echo -e "${YELLOW}⚠️  Error creating vector store, skipping...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  File not found, skipping${NC}"
fi

echo ""
echo -e "${BLUE}Step 4: Training IsolationForest model...${NC}"
if [ -f "src/model_training.py" ]; then
    python src/model_training.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model training complete${NC}"
    else
        echo -e "${YELLOW}⚠️  Error in model training, skipping...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  File not found, skipping${NC}"
fi

echo ""
echo -e "${BLUE}Step 5: Evaluating model...${NC}"
if [ -f "src/evaluate_model.py" ]; then
    python src/evaluate_model.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model evaluation complete${NC}"
    else
        echo -e "${YELLOW}⚠️  Error in evaluation, skipping...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  File not found, skipping${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo -e "✓ All steps complete!"
echo -e "==========================================${NC}"
echo ""
echo -e "${BLUE}Step 6: Launching Streamlit App...${NC}"
echo -e "${YELLOW}Starting Streamlit...${NC}"
echo ""
echo -e "${BLUE}App will open at: http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""
sleep 2

streamlit run src/app.py

