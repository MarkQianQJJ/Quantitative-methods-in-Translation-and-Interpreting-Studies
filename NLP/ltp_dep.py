"""LTP Demo - 整理汉语句子的依存关系和依存方向"""

import os
import torch
import pandas as pd
from ltp import LTP

ltp = LTP(os.path.join(r"D:\vscode\corpus data\ltp_models", "small"))
if torch.cuda.is_available():
    ltp.to("cuda")

ltp.add_word("汤姆", freq=2)
ltp.add_words(["外套", "外衣"], freq=2)

sentences = ["他叫汤姆去拿外衣。"]
output = ltp.pipeline(sentences, tasks=["cws", "pos", "dep"])

data = []
for sent_id, sentence in enumerate(sentences, 1):
    words = output.cws[sent_id - 1]
    pos_tags = output.pos[sent_id - 1]
    dep_item = output.dep[sent_id - 1]

    heads = dep_item["head"]
    dep_labels = dep_item["label"]

    for idx, token in enumerate(words, 1):
        head_id = heads[idx - 1]
        dep = dep_labels[idx - 1]
        pos = pos_tags[idx - 1]

        if head_id == 0:
            head_text = "ROOT"
            head_pos = "ROOT"
        else:
            head_text = words[head_id - 1]
            head_pos = pos_tags[head_id - 1]

        if head_id == 0 or dep == "HED" or dep == "WP":
            ddir = "/"
            dd = "/"
        else:
            ddir = "head_final" if head_id > idx else "head_initial"
            dd = abs(idx - head_id)

        data.append({
            "Sentence_ID": sent_id,
            "Token_ID": idx,
            "token": token,
            "pos": pos,
            "dep": dep,
            "head_text": head_text,
            "head_ID": head_id,
            "head_pos": head_pos,
            "Ddir": ddir,
            "DD": dd
        })

df = pd.DataFrame(data)
output_path = r"D:\vscode\corpus data\ltp_models\chinese_tokens_analysis_with_Ddir_DD_demo.xlsx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_excel(output_path, index=False)

print(df)
print(f"Excel 文件已保存到: {output_path}")
