# -*- coding: utf-8 -*-
"""
英译汉 COMET 评测示例
模型：Unbabel/wmt22-comet-da（已下载为本地 ckpt）
"""

from comet import load_from_checkpoint

# 1. 加载 COMET 模型（你本地的 ckpt 路径）
model_path = r"D:\vscode\corpus data\comet_model\checkpoints\model.ckpt"
model = load_from_checkpoint(model_path)

# 2. 准备英→中数据
# src: 英文原文
# mt : 机器翻译的中文
# ref: 参考译文（人工翻译）的中文
data = [
    {
        "src": "I like to eat apples.",
        "mt":  "我很喜欢吃苹果。",
        "ref": "我喜欢吃苹果。"
    },
    {
        "src": "The weather is great, let's go to the park.",
        "mt":  "天气很好，我们去公园吧。",
        "ref": "今天天气不错，我们去公园吧。"
    }
]

# 3. 运行评测
# 如果你有 GPU，gpus=1；如果没有，改成 gpus=0 或去掉 gpus 参数
output = model.predict(data, batch_size=8, gpus=1)

# 按你现在用的方式取结果（老版本 COMET 通常是这样）
seg_scores = output[0]   # 每个句子的分数
sys_score = output[1]    # 系统整体分数（平均）

# 4. 打印结果
print("Segment-level COMET scores:", seg_scores)
print("System-level COMET score:", sys_score)
