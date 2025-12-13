#https://www.nltk.org/howto/bleu.html
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

references = ['The quick brown fox leaps over the lazy dog.'.split()]
hypothesis = 'The quick brown Mary.'.split()

score = sentence_bleu(references, hypothesis, smoothing_function=SmoothingFunction().method4)
print(f"BLEU score: {score * 100:.2f}")



from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Step 1: Define references and hypothesis
references = [
    ['the', 'cat', 'is', 'on', 'the', 'mat'],
    ['there', 'is', 'a', 'cat', 'on', 'the', 'mat']
]

hypothesis = ['the', 'cat', 'the', 'cat', 'on', 'the', 'mat']

# Step 2: Compute BLEU-1 (unigram only)
bleu1_score = sentence_bleu(
    references,
    hypothesis,
    weights=(1, 0, 0, 0),  # unigram only
    smoothing_function=SmoothingFunction().method1  # optional smoothing
)

print("BLEU-1 score:", bleu1_score)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

references = [
    ['the', 'cat', 'is', 'on', 'the', 'mat'],
    ['there', 'is', 'a', 'cat', 'on', 'the', 'mat']
]

hypothesis = ['the', 'cat', 'the', 'cat', 'on', 'the', 'mat']

bleu4_score = sentence_bleu(
    references,
    hypothesis,
    weights=(0.25, 0.25, 0.25, 0.25),  # equal weights for 1-gram to 4-gram
    smoothing_function=SmoothingFunction().method1  # recommended smoothing
)

print("BLEU-4 score:", bleu4_score)

