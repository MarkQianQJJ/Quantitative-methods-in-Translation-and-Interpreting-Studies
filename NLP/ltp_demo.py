"""LTP Demo - 使用本地神经网络模型"""
#https://github.com/HIT-SCIR/ltp/blob/main/python/interface/docs/quickstart.rst
#https://huggingface.co/LTP/small/tree/main
#https://ltp.readthedocs.io/zh-cn/latest/appendix.html

import os
import torch
from ltp import LTP

ltp = LTP(os.path.join(r"D:\vscode\corpus data\ltp_models", "small"))

if torch.cuda.is_available():
    ltp.to("cuda")

ltp.add_word("汤姆", freq=2)
ltp.add_words(["外套", "外衣"], freq=2)

output = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])

print("分词:", output.cws)
print("词性:", output.pos)
print("命名实体:", output.ner)
print("语义角色标注:", output.srl)
print("依存句法:", output.dep)
print("语义依存分析树:", output.sdp)
print("语义依存分析图:", output.sdpg)