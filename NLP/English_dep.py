import spacy
import pandas as pd

# 加载spaCy的英文模型
nlp = spacy.load("en_core_web_sm")

# 需要分析的句子
doc = nlp("Autonomous cars shift insurance liability toward manufacturers.")

# 准备存储数据
data = []

# 遍历每个token（单词）
for idx, token in enumerate(doc, 1):  # Token_ID从1开始
    head_id = [i+1 for i, t in enumerate(doc) if t.text == token.head.text][0]  # 查找head_token的ID
    
    # 计算 Ddir 列
    if token.dep_ == "ROOT" or token.dep_ == "punct":
        ddir = "/"
        dd = "/"
    else:
        ddir = "head_final" if head_id > idx else "head_initial"
        dd = abs(idx - head_id)
    
    data.append({
        'Token_ID': idx,
        'token': token.text,
        'dep': token.dep_,
        'head_text': token.head.text,
        'head_ID': head_id,
        'head_pos': token.head.pos_,
        'children': [child.text for child in token.children],
        'Ddir': ddir,
        'DD': dd
    })

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 保存为Excel文件
output_path = r"D:\vscode\corpus data\Wenchao_data\data\tokens_analysis_with_Ddir_DD_demo.xlsx"
df.to_excel(output_path, index=False)

print(f"Excel 文件已保存到: {output_path}")