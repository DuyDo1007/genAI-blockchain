"""
Sinh dữ liệu synthetic/fraud smart contracts bằng Generative AI
Hỗ trợ OpenAI API hoặc local models
"""
import os
import pandas as pd
import json
from typing import List, Dict


OUT_CSV = 'data/synthetic/gen_synthetic_findings.csv'


# Option A: OpenAI API (khuyến nghị)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI không được cài đặt. Sử dụng local model.")


# Option B: Local model với transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers không được cài đặt.")


PROMPT_TEMPLATE = """Generate a realistic smart contract security finding in JSON format with the following structure:
{
  "title": "Short title of the vulnerability",
  "content": "Detailed description of the vulnerability (2-4 sentences). Include contract name, function name, and code snippet if relevant.",
  "impact": "LOW or MEDIUM or HIGH",
  "contract_name": "Name of the contract",
  "function_name": "Name of the vulnerable function",
  "code": "Code snippet showing the vulnerability",
  "vulnerability_label": "Type of vulnerability (e.g., REENTRANCY, OVERFLOW, ACCESS_CONTROL)"
}

Focus on common smart contract vulnerabilities like:
- Reentrancy attacks
- Integer overflow/underflow
- Access control issues
- Unchecked external calls
- Front-running vulnerabilities
- Gas optimization issues

Generate only the JSON object, no additional text."""


def generate_with_openai(n=50, model="gpt-3.5-turbo"):
    """
    Sinh dữ liệu synthetic bằng OpenAI API
    
    Args:
        n: Số lượng samples cần sinh
        model: Model name (gpt-3.5-turbo hoặc gpt-4)
    
    Returns:
        List[Dict]: Danh sách các findings đã sinh
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package không được cài đặt")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable chưa được set")
    
    client = OpenAI(api_key=api_key)
    outputs = []
    
    print(f"Đang sinh {n} samples bằng OpenAI {model}...")
    
    for i in range(n):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in smart contract security."},
                    {"role": "user", "content": PROMPT_TEMPLATE}
                ],
                temperature=0.8,
                max_tokens=500
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Parse JSON từ response
            try:
                # Loại bỏ markdown code blocks nếu có
                if generated_text.startswith('```'):
                    generated_text = generated_text.split('```')[1]
                    if generated_text.startswith('json'):
                        generated_text = generated_text[4:]
                
                finding = json.loads(generated_text)
                finding['id'] = 90000 + i  # Synthetic ID
                finding['protocol_id'] = 9999
                finding['auditfirm_id'] = 99
                outputs.append(finding)
            except json.JSONDecodeError:
                print(f"⚠️  Không parse được JSON từ sample {i+1}, bỏ qua")
                continue
                
        except Exception as e:
            print(f"⚠️  Lỗi khi sinh sample {i+1}: {e}")
            continue
    
    print(f"✓ Đã sinh {len(outputs)} samples thành công")
    return outputs


def generate_local(n=50):
    """
    Sinh dữ liệu synthetic bằng local model (GPT-2)
    
    Args:
        n: Số lượng samples cần sinh
    
    Returns:
        List[Dict]: Danh sách các findings đã sinh
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers package không được cài đặt")
    
    print(f"Đang tải model GPT-2...")
    gen = pipeline('text-generation', model='gpt2', device=-1)
    
    outputs = []
    print(f"Đang sinh {n} samples bằng local model...")
    
    for i in range(n):
        try:
            prompt = f"Smart contract vulnerability finding: {PROMPT_TEMPLATE[:100]}"
            o = gen(prompt, max_length=300, num_return_sequences=1, 
                   temperature=0.8, do_sample=True)[0]['generated_text']
            
            # Tạo structured data từ generated text
            finding = {
                'id': 90000 + i,
                'title': f"Synthetic Vulnerability {i+1}",
                'content': o[len(prompt):].strip()[:500],
                'impact': ['LOW', 'MEDIUM', 'HIGH'][i % 3],
                'contract_name': f"Contract_{i+1}",
                'function_name': f"function_{i+1}",
                'code': f"// Synthetic code snippet {i+1}",
                'vulnerability_label': ['REENTRANCY', 'OVERFLOW', 'ACCESS_CONTROL'][i % 3],
                'protocol_id': 9999,
                'auditfirm_id': 99
            }
            outputs.append(finding)
        except Exception as e:
            print(f"⚠️  Lỗi khi sinh sample {i+1}: {e}")
            continue
    
    print(f"✓ Đã sinh {len(outputs)} samples thành công")
    return outputs


def generate_synthetic_data(n=50, use_openai=True, model="gpt-3.5-turbo"):
    """
    Sinh dữ liệu synthetic smart contracts
    
    Args:
        n: Số lượng samples
        use_openai: Sử dụng OpenAI API nếu True, local model nếu False
        model: Model name cho OpenAI
    
    Returns:
        DataFrame chứa synthetic findings
    """
    if use_openai and OPENAI_AVAILABLE:
        outputs = generate_with_openai(n, model)
    elif TRANSFORMERS_AVAILABLE:
        print("⚠️  Sử dụng local model (chất lượng thấp hơn OpenAI)")
        outputs = generate_local(n)
    else:
        raise ImportError("Cần cài đặt OpenAI hoặc transformers package")
    
    if not outputs:
        raise ValueError("Không sinh được samples nào")
    
    df = pd.DataFrame(outputs)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8')
    print(f"✓ Đã lưu {len(df)} synthetic samples vào {OUT_CSV}")
    
    return df


if __name__ == '__main__':
    # Thử OpenAI trước, nếu không có thì dùng local
    use_openai = OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY') is not None
    
    if use_openai:
        print("Sử dụng OpenAI API...")
        generate_synthetic_data(n=50, use_openai=True)
    else:
        print("Sử dụng local model (GPT-2)...")
        generate_synthetic_data(n=50, use_openai=False)