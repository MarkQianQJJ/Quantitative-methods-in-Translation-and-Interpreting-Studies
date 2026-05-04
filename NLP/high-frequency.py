import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string

nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords

def load_high_frequency_words(filepath):
    """Load high frequency words from file where each line contains multiple words"""
    with open(filepath, 'r') as f:
        words = set()
        for line in f:
            words.update(word.strip() for word in line.split())
        return words

def get_high_frequency_proportion(text, high_freq_words, remove_stopwords=True):
    """Calculate proportion of high frequency words in text"""
    # 1. Tokenize words
    words = word_tokenize(text.lower())
    
    # 2. Remove punctuation and non-alphabetic words
    words = [word for word in words if word.isalpha()]
    
    # 3. Remove stopwords (optional)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    
    # 4. Count high frequency words
    high_freq_count = sum(1 for word in words if word in high_freq_words)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0, []
    
    proportion = high_freq_count / total_words
    matching_words = [word for word in words if word in high_freq_words]
    
    return proportion, matching_words

# Load high frequency words
high_freq_words = load_high_frequency_words('coca_heads(01)_expanded.txt')

# Example usage
text = "It gathered 53 authoritative experts and took a year to produce. The blue book argues that with AI empowerment, research in the humanities and social sciences is entering a fifth paradigm driven by both data and mechanisms, which will accelerate innovation in new liberal arts."
proportion, matching_words = get_high_frequency_proportion(text, high_freq_words)

print(f"High frequency word proportion: {proportion:.2%}")
print(f"Matching high frequency words: {set(matching_words)}")
