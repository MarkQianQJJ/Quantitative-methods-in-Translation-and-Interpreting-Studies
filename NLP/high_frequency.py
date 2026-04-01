import string

def load_high_frequency_words(filepath):
    """Load high frequency words from file where each line contains multiple words"""
    with open(filepath, 'r') as f:
        words = set()
        for line in f:
            words.update(word.strip() for word in line.split())
        return words

def get_high_frequency_proportion(text, high_freq_words):
    """Calculate proportion of high frequency words in text"""
    # 1. Convert text to lowercase and split into words (tokenization)
    text = text.lower()
    
    # 2. Remove punctuation and non-alphabetic words
    words = [word.strip(string.punctuation) for word in text.split() if word.isalpha()]
    
    # 3. Count high frequency words
    high_freq_count = sum(1 for word in words if word in high_freq_words)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0, []
    
    # 4. Calculate high frequency word proportion
    proportion = high_freq_count / total_words
    matching_words = [word for word in words if word in high_freq_words]
    
    return proportion, matching_words

# Load high frequency words from file
high_freq_words = load_high_frequency_words('corpus data/coca_heads(01)_expanded.txt')

# Example usage
text = "It gathered 53 authoritative experts and took a year to produce. The blue book argues that with AI empowerment, research in the humanities and social sciences is entering a fifth paradigm driven by both data and mechanisms, which will accelerate innovation in new liberal arts."
proportion, matching_words = get_high_frequency_proportion(text, high_freq_words)

print(f"High frequency word proportion: {proportion:.2%}")
print(f"Matching high frequency words: {set(matching_words)}")