"""
Xử lý dữ liệu: Chuyển đổi JSON files sang CSV
Trích xuất các trường quan trọng từ smart contract findings
Sử dụng LLM để tách code chính xác hơn

QUAN TRỌNG: Chỉ lưu các dòng có code vào CSV để sử dụng cho training.
Các dòng không có code sẽ bị bỏ qua.
"""
import os
import sys
import io
import json
import pandas as pd
from glob import glob
import re
from typing import Tuple, Optional
import hashlib

def _get_api_key_from_file() -> Optional[str]:
    """
    Thử đọc API key từ file .env hoặc config.json
    """
    # Thử đọc từ .env file
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Xử lý cả trường hợp có khoảng trắng: OPENAI_API_KEY = value hoặc OPENAI_API_KEY=value
                    if 'OPENAI_API_KEY' in line and '=' in line:
                        # Tách theo dấu = và lấy phần sau
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            key_part = parts[0].strip()
                            if key_part == 'OPENAI_API_KEY':
                                value = parts[1].strip().strip('"').strip("'")
                                return value
        except Exception as e:
            pass
    
    # Thử đọc từ config.json
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('openai_api_key') or config.get('OPENAI_API_KEY')
        except Exception:
            pass
    
    return None

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Thử import OpenAI, nếu không có thì dùng fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI không khả dụng, sẽ sử dụng phương pháp regex fallback")

RAW_DIR = 'data/raw'
OUT_CSV = 'data/processed/findings.csv'

# Cache để tránh gọi LLM nhiều lần cho cùng một content
_extraction_cache = {}
_api_key_warning_shown = False  # Để chỉ hiển thị warning một lần
_llm_error_shown = False  # Để chỉ hiển thị lỗi LLM một lần
_llm_disabled = False  # Flag để disable LLM sau khi gặp lỗi nghiêm trọng (quota, rate limit, etc.)


def extract_code_with_llm(content: str, use_cache: bool = True) -> Tuple[str, str, str]:
    """
    Sử dụng LLM để trích xuất code, contract_name và function_name từ content
    
    Args:
        content: Nội dung text cần phân tích
        use_cache: Có sử dụng cache hay không
    
    Returns:
        Tuple (contract_name, function_name, code)
    """
    global _api_key_warning_shown, _llm_error_shown, _llm_disabled
    
    # Nếu LLM đã bị disable (do lỗi quota/rate limit), dùng fallback ngay
    if _llm_disabled:
        return extract_contract_info_fallback(content)
    
    if not content or not OPENAI_AVAILABLE:
        return extract_contract_info_fallback(content)
    
    # Kiểm tra cache
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    if use_cache and content_hash in _extraction_cache:
        return _extraction_cache[content_hash]
    
    try:
        # Lấy API key từ nhiều nguồn
        api_key = (
            os.getenv('OPENAI_API_KEY') or 
            os.getenv('OPENAI_KEY') or
            _get_api_key_from_file()
        )
        
        if not api_key:
            if not _api_key_warning_shown:
                print("⚠️  OPENAI_API_KEY không được thiết lập, sử dụng fallback")
                print("   💡 Để sử dụng LLM, thiết lập biến môi trường: export OPENAI_API_KEY='your-key'")
                _api_key_warning_shown = True
            return extract_contract_info_fallback(content)
        
        client = OpenAI(api_key=api_key)
        
        # Giới hạn content để tránh vượt quá token limit (giữ lại 8000 ký tự đầu)
        content_truncated = content[:8000] if len(content) > 8000 else content
        
        prompt = f"""Phân tích nội dung sau và trích xuất thông tin về smart contract code. 
Nếu có code Solidity hoặc code liên quan, hãy trích xuất:
1. Contract name (tên contract chính, ví dụ: SVFHook, CredibleAccountModule)
2. Function name (tên function chính được đề cập, ví dụ: addLiquidity, configure)
3. Code blocks (tất cả các đoạn code trong markdown code blocks ```)

Nếu KHÔNG có code, trả về JSON với các giá trị rỗng.

Trả về JSON format:
{{
    "contract_name": "tên contract hoặc rỗng",
    "function_name": "tên function hoặc rỗng", 
    "code": "toàn bộ code blocks nối lại bằng \\n\\n hoặc rỗng nếu không có code"
}}

Nội dung cần phân tích:
{content_truncated}

Chỉ trả về JSON, không có text thêm."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Sử dụng model rẻ hơn để tiết kiệm chi phí
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia phân tích smart contract code. Trả về JSON chính xác."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Loại bỏ markdown code blocks nếu có
        if result_text.startswith('```'):
            result_text = re.sub(r'^```(?:json)?\s*\n', '', result_text)
            result_text = re.sub(r'\n```\s*$', '', result_text)
        
        result = json.loads(result_text)
        contract_name = result.get('contract_name', '').strip()
        function_name = result.get('function_name', '').strip()
        code = result.get('code', '').strip()
        
        # Lưu vào cache
        if use_cache:
            _extraction_cache[content_hash] = (contract_name, function_name, code)
        
        return contract_name, function_name, code
        
    except json.JSONDecodeError as e:
        if not _llm_error_shown:
            print(f"⚠️  Lỗi parse JSON từ LLM: {e}")
            _llm_error_shown = True
        return extract_contract_info_fallback(content)
    except Exception as e:
        error_msg = str(e)
        
        # Kiểm tra các lỗi nghiêm trọng cần disable LLM ngay
        is_critical_error = (
            '429' in error_msg or  # Rate limit / Quota exceeded
            'insufficient_quota' in error_msg.lower() or
            'quota' in error_msg.lower() or
            'rate_limit' in error_msg.lower() or
            'too_many_requests' in error_msg.lower()
        )
        
        if is_critical_error:
            # Disable LLM cho tất cả các request tiếp theo
            _llm_disabled = True
            if not _llm_error_shown:
                print(f"⚠️  Lỗi nghiêm trọng khi gọi LLM: {error_msg}")
                print("   → Đã vượt quá quota/rate limit của OpenAI API")
                print("   → Tự động chuyển sang phương pháp regex fallback cho tất cả files còn lại...")
                print("   💡 Để sử dụng LLM, vui lòng:")
                print("      - Kiểm tra quota tại: https://platform.openai.com/account/billing")
                print("      - Hoặc đợi một lúc rồi thử lại")
                _llm_error_shown = True
        elif not _llm_error_shown:
            print(f"⚠️  Lỗi khi gọi LLM: {error_msg}")
            # Kiểm tra nếu là lỗi API key
            if '401' in error_msg or 'invalid_api_key' in error_msg or 'Incorrect API key' in error_msg:
                print("   💡 Vui lòng kiểm tra lại OPENAI_API_KEY trong file .env")
                print("   💡 OpenAI API key thường bắt đầu bằng 'sk-'")
                print("   💡 Lấy API key tại: https://platform.openai.com/account/api-keys")
            print("   → Đang sử dụng phương pháp regex fallback cho file này...")
            _llm_error_shown = True
        
        return extract_contract_info_fallback(content)


def extract_contract_info_fallback(content: str) -> Tuple[str, str, str]:
    """
    Phương pháp fallback sử dụng regex để trích xuất thông tin
    """
    contract_name = ''
    function_name = ''
    code = ''
    
    if not content:
        return contract_name, function_name, code
    
    # Tìm contract name trong content (pattern: `ContractName` hoặc "ContractName")
    contract_match = re.search(r'`([A-Z][a-zA-Z0-9_]+)`|"([A-Z][a-zA-Z0-9_]+)"', content)
    if contract_match:
        contract_name = contract_match.group(1) or contract_match.group(2) or ''
    
    # Tìm function name (pattern: `functionName()` hoặc function functionName)
    func_match = re.search(r'`([a-z][a-zA-Z0-9_]+)\(\)`|function\s+([a-z][a-zA-Z0-9_]+)', content, re.IGNORECASE)
    if func_match:
        function_name = func_match.group(1) or func_match.group(2) or ''
    
    # Trích xuất code blocks nếu có - ưu tiên các block có ngôn ngữ (solidity, javascript, etc)
    # Pattern 1: Code blocks có ngôn ngữ được chỉ định
    code_blocks_with_lang = re.findall(r'```(?:solidity|javascript|typescript|python|rust|go|java|cpp|c\+\+|c|shell|bash)\s*\n([\s\S]*?)```', content, re.IGNORECASE)
    # Pattern 2: Tất cả code blocks (fallback)
    all_code_blocks = re.findall(r'```[^\n]*\n([\s\S]*?)```', content)
    
    # Ưu tiên code blocks có ngôn ngữ, nếu không có thì dùng tất cả
    code_blocks = code_blocks_with_lang if code_blocks_with_lang else all_code_blocks
    
    if code_blocks:
        # Lọc và làm sạch code blocks
        cleaned_blocks = []
        for block in code_blocks[:5]:  # Lấy tối đa 5 code blocks
            cleaned = block.strip()
            # Chỉ lấy block có độ dài hợp lý (ít nhất 20 ký tự, không quá 5000 ký tự)
            if 20 <= len(cleaned) <= 5000:
                cleaned_blocks.append(cleaned)
        if cleaned_blocks:
            code = '\n\n---\n\n'.join(cleaned_blocks)
    
    return contract_name, function_name, code


def extract_contract_info(content: str, use_llm: bool = True, use_cache: bool = True) -> Tuple[str, str, str]:
    """
    Trích xuất contract_name, function_name và code từ content
    Sử dụng LLM nếu có, nếu không thì dùng regex fallback
    
    Args:
        content: Nội dung text cần phân tích
        use_llm: Có sử dụng LLM hay không (mặc định True)
        use_cache: Có sử dụng cache hay không (mặc định True)
    
    Returns:
        Tuple (contract_name, function_name, code)
    """
    global _llm_disabled
    
    # Nếu LLM đã bị disable hoặc không khả dụng, dùng fallback
    if use_llm and OPENAI_AVAILABLE and not _llm_disabled:
        return extract_code_with_llm(content, use_cache)
    else:
        return extract_contract_info_fallback(content)


def jsons_to_csv(raw_dir=RAW_DIR, out_csv=OUT_CSV, use_llm: bool = True, use_cache: bool = True):
    """
    Đọc tất cả JSON files trong raw_dir và chuyển sang CSV
    Sử dụng LLM để tách code chính xác hơn
    
    QUAN TRỌNG: Chỉ lưu các dòng có code vào CSV. Các dòng không có code sẽ bị bỏ qua.
    
    Args:
        raw_dir: Thư mục chứa JSON files
        out_csv: Đường dẫn file CSV output
        use_llm: Có sử dụng LLM để tách code hay không (mặc định True - KHUYẾN NGHỊ)
        use_cache: Có sử dụng cache để tránh gọi LLM nhiều lần (mặc định True)
    
    Returns:
        DataFrame chứa dữ liệu đã xử lý (chỉ các dòng có code)
    """
    rows = []
    files = glob(os.path.join(raw_dir, '*.json'))
    
    print(f"Đang xử lý {len(files)} file JSON...")
    if use_llm and OPENAI_AVAILABLE:
        print("✓ Sử dụng LLM để tách code chính xác hơn")
    else:
        print("⚠️  Sử dụng phương pháp regex fallback")
    
    processed = 0
    for p in files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                j = json.load(f)
        except Exception as e:
            print(f"Lỗi khi đọc file {p}: {e}")
            continue
        
        # Trích xuất contract info từ content
        content = j.get('content', '')
        contract_name, function_name, code = extract_contract_info(content, use_llm=use_llm, use_cache=use_cache)
        
        # CHỈ LƯU CÁC DÒNG CÓ CODE (code không rỗng)
        # Loại bỏ các khoảng trắng và kiểm tra xem có code thực sự không
        code_cleaned = code.strip() if code else ''
        
        if not code_cleaned:
            # Bỏ qua dòng này nếu không có code
            processed += 1
            if processed % 50 == 0:
                print(f"  Đã xử lý {processed}/{len(files)} files...")
            continue
        
        # Lấy vulnerability label từ impact hoặc title
        impact = j.get('impact', '')
        vulnerability_label = impact if impact else 'UNKNOWN'
        
        # Lấy các trường quan trọng
        row = {
            'id': j.get('id'),
            'title': j.get('title', ''),
            'content': content,
            'impact': impact,
            'protocol_id': j.get('protocol_id'),
            'auditfirm_id': j.get('auditfirm_id'),
            'contract_name': contract_name,
            'function_name': function_name,
            'code': code_cleaned,
            'vulnerability_label': vulnerability_label,
            'protocol_name': j.get('protocol_name', ''),
            'firm_name': j.get('firm_name', '')
        }
        rows.append(row)
        
        processed += 1
        if processed % 50 == 0:
            print(f"  Đã xử lý {processed}/{len(files)} files...")
    
    df = pd.DataFrame(rows)
    
    # Thống kê trước khi lọc
    total_processed = processed
    total_with_code = len(df)
    skipped = total_processed - total_with_code
    
    if len(df) == 0:
        print(f"\n⚠️  CẢNH BÁO: Không có dòng nào có code sau khi xử lý!")
        print(f"   Đã xử lý {total_processed} files, nhưng không tìm thấy code trong bất kỳ file nào.")
        print(f"   Vui lòng kiểm tra:")
        print(f"   - OPENAI_API_KEY đã được thiết lập chưa?")
        print(f"   - Các file JSON có chứa code blocks không?")
        return df
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"\n✓ Đã ghi {len(df)} dòng CÓ CODE vào {out_csv}")
    print(f"  - Tổng số files đã xử lý: {total_processed}")
    print(f"  - Files có code: {total_with_code} ({total_with_code/total_processed*100:.1f}%)")
    print(f"  - Files bỏ qua (không có code): {skipped} ({skipped/total_processed*100:.1f}%)")
    print(f"  - Có contract_name: {df['contract_name'].notna().sum()} ({df['contract_name'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  - Có function_name: {df['function_name'].notna().sum()} ({df['function_name'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  - Tất cả dòng đều có code: {df['code'].notna().sum()} ({df['code'].notna().sum()/len(df)*100:.1f}%)")
    
    return df


if __name__ == '__main__':
    jsons_to_csv()