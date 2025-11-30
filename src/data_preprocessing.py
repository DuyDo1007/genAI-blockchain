"""
Xử lý dữ liệu: Chuyển đổi JSON files sang CSV
Trích xuất các trường quan trọng từ smart contract findings
"""
import os
import sys
import io
import json
import pandas as pd
from glob import glob
import re

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


RAW_DIR = 'data/raw'
OUT_CSV = 'data/processed/findings.csv'


def extract_contract_info(content):
    """
    Trích xuất contract_name và function_name từ content nếu có
    """
    contract_name = ''
    function_name = ''
    code = ''
    
    if content:
        # Tìm contract name trong content (pattern: `ContractName` hoặc "ContractName")
        contract_match = re.search(r'`([A-Z][a-zA-Z0-9_]+)`|"([A-Z][a-zA-Z0-9_]+)"', content)
        if contract_match:
            contract_name = contract_match.group(1) or contract_match.group(2) or ''
        
        # Tìm function name (pattern: `functionName()` hoặc function functionName)
        func_match = re.search(r'`([a-z][a-zA-Z0-9_]+)\(\)`|function\s+([a-z][a-zA-Z0-9_]+)', content, re.IGNORECASE)
        if func_match:
            function_name = func_match.group(1) or func_match.group(2) or ''
        
        # Trích xuất code blocks nếu có
        code_blocks = re.findall(r'```[\s\S]*?```|`[^`]+`', content)
        if code_blocks:
            code = '\n'.join(code_blocks[:3])  # Lấy tối đa 3 code blocks
    
    return contract_name, function_name, code


def jsons_to_csv(raw_dir=RAW_DIR, out_csv=OUT_CSV):
    """
    Đọc tất cả JSON files trong raw_dir và chuyển sang CSV
    
    Args:
        raw_dir: Thư mục chứa JSON files
        out_csv: Đường dẫn file CSV output
    
    Returns:
        DataFrame chứa dữ liệu đã xử lý
    """
    rows = []
    files = glob(os.path.join(raw_dir, '*.json'))
    
    print(f"Đang xử lý {len(files)} file JSON...")
    
    for p in files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                j = json.load(f)
        except Exception as e:
            print(f"Lỗi khi đọc file {p}: {e}")
            continue
        
        # Trích xuất contract info từ content
        content = j.get('content', '')
        contract_name, function_name, code = extract_contract_info(content)
        
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
            'code': code,
            'vulnerability_label': vulnerability_label,
            'protocol_name': j.get('protocol_name', ''),
            'firm_name': j.get('firm_name', '')
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"✓ Đã ghi {len(df)} dòng vào {out_csv}")
    print(f"  - Có contract_name: {df['contract_name'].notna().sum()}")
    print(f"  - Có function_name: {df['function_name'].notna().sum()}")
    print(f"  - Có code: {df['code'].notna().sum()}")
    
    return df


if __name__ == '__main__':
    jsons_to_csv()