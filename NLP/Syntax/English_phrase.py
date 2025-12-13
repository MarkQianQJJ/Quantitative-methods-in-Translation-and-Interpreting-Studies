import nltk
from stanfordcorenlp import StanfordCoreNLP

# Function to extract NP, VP, and PP from constituency parse tree
def extract_phrases_from_tree(parse_tree):
    np_list, vp_list, pp_list = [], [], []
    
    if isinstance(parse_tree, nltk.Tree):
        for subtree in parse_tree:
            if isinstance(subtree, nltk.Tree):
                label = subtree.label()
                if label == 'NP':
                    np_list.append(" ".join(subtree.leaves()))
                elif label == 'VP':
                    vp_list.append(" ".join(subtree.leaves()))
                elif label == 'PP':
                    pp_list.append(" ".join(subtree.leaves()))
                # Recursively extract phrases from subtrees
                sub_np, sub_vp, sub_pp = extract_phrases_from_tree(subtree)
                np_list.extend(sub_np)
                vp_list.extend(sub_vp)
                pp_list.extend(sub_pp)
    
    return np_list, vp_list, pp_list

# Connect to the CoreNLP server
#D:\vscode\stanford-corenlp-4.5.8>java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9999 -timeout 15000
nlp = StanfordCoreNLP('http://localhost', port=9999)

# Sample sentence for testing
sentence = "The quick brown fox jumped over the lazy dog."

# Constituency Parsing
constituency_parse = nlp.parse(sentence)

# Parse the constituency tree
try:
    parse_tree = nltk.Tree.fromstring(constituency_parse)
except Exception as e:
    print(f"Error in parsing tree: {e}")

# Extract NP, VP, and PP
np_list, vp_list, pp_list = extract_phrases_from_tree(parse_tree)

# Print results
print(f"Extracted NP List: {np_list}")
print(f"Extracted VP List: {vp_list}")
print(f"Extracted PP List: {pp_list}")

# Close the connection
nlp.close()