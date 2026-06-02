import os
import time
import requests
from pathlib import Path

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-afbf53565a0a41498fe2a57e7ca62623"  # 记得使用后删除

def translate_text(text):
    """翻译英文文本到中文"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Translate English to natural, fluent Chinese"},
            {"role": "user", "content": text}
        ],
        "temperature": 1.3
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def process_files():
    input_dir = Path(r"D:\vscode\corpus data\deepseek\raw")
    output_dir = Path(r"D:\vscode\corpus data\deepseek\translated")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt') and not filename.startswith('ZH_'):
            input_path = input_dir / filename
            output_filename = f"ZH_{filename}" if not filename.startswith('EN_') else filename.replace('EN_', 'ZH_')
            output_path = output_dir / output_filename
            
            if output_path.exists():
                print(f"跳过 {filename} - 已翻译")
                continue
                
            print(f"处理 {filename}...")
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 简单分块处理（每1000字符）
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            translated_text = ""
            
            for chunk in chunks:
                translated_chunk = translate_text(chunk)
                translated_text += translated_chunk
                time.sleep(1)  # 避免请求过快
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            
            print(f"完成 {filename} 的翻译")

if __name__ == "__main__":
    process_files()
    print("处理完成！请记得删除API密钥")