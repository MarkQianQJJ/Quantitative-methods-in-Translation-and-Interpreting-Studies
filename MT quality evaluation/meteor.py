#https://www.nltk.org/howto/meteor.html
# pip install nltk
import nltk
from nltk.translate import meteor
from nltk import word_tokenize

# 参考译文（可多个）和候选译文（都先分词）
references = [word_tokenize('The cat sat on the mat')]
hypothesis  = word_tokenize('The cat was sat on the mat')

score = meteor(references, hypothesis)   # 0~1 之间
print(round(score, 4))                   # 例：0.9654