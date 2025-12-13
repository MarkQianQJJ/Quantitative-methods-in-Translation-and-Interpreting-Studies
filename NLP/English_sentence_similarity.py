import spacy

# 加载模型
nlp = spacy.load('en_core_web_lg')

# 语料库
corpus = [
    'this camera is perfect for an enthusiastic amateur photographer',
    'it is light enough to carry around all day without bother',
    'i love photography',
    'the speed is noticeably slower than canon, especially so with flashes on',
    'be very careful when the battery is low and make sure to carry extra batteries',
    'i enthusiastically recommend this camera',
    'you have to manually take the cap off in order to use it'
]

# 计算相似度
l1 = []
l2 = []
l3 = []

for line1 in corpus:
    sent1 = nlp(line1)
    for line2 in corpus:
        sent2 = nlp(line2)
        if line1 != line2:
            l1.append(line1)
            l2.append(line2)
            l3.append(sent1.similarity(sent2))

# 组合结果
pair_list = zip(l3, l1, l2)

# 排序
sorted_data = sorted(pair_list, key=lambda result: result[0], reverse=True)

# 去除重复对（每隔一个取一个）
dataList = []
for n in range(0, len(sorted_data), 2):
    dataList.append(sorted_data[n])

# 输出结果
print("句子相似度计算结果（按相似度降序排列，已去重）：")
for similarity, sent1, sent2 in dataList:
    print(f"相似度: {similarity:.4f}")
    print(f"句子1: {sent1}")
    print(f"句子2: {sent2}")
    print("-" * 50)