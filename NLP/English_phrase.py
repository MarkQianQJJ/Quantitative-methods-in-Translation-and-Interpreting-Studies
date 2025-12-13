import nltk
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP

# Function to extract NP, VP, PP, and coordinate phrases from constituency parse tree
def extract_phrases_from_tree(parse_tree):
    np_list, vp_list, pp_list, coord_list = [], [], [], []
    
    if isinstance(parse_tree, nltk.Tree):
        for subtree in parse_tree:
            if isinstance(subtree, nltk.Tree):
                label = subtree.label()
                
                # Capture Noun Phrases (NP)
                if label == 'NP':
                    np_list.append(" ".join(subtree.leaves()))
                
                # Capture Verb Phrases (VP)
                elif label == 'VP':
                    vp_list.append(" ".join(subtree.leaves()))
                
                # Capture Prepositional Phrases (PP)
                elif label == 'PP':
                    pp_list.append(" ".join(subtree.leaves()))
                
                # Look for Coordinating Conjunction (CC) and extract the coordinated phrases
                elif label == 'CC':  # Coordinating conjunction (like "and", "or")
                    parent = parse_tree  # Start at the root and iterate over the parent structure
                    left_phrase = None
                    right_phrase = None
                    
                    # Iterate over the parent to get the left and right subtrees of the CC
                    for i, sibling in enumerate(parent):
                        if isinstance(sibling, nltk.Tree) and sibling != subtree:
                            if left_phrase is None:
                                left_phrase = " ".join(sibling.leaves())  # Left phrase
                            else:
                                right_phrase = " ".join(sibling.leaves())  # Right phrase
                    
                    # If both left and right phrases are found, combine them as one coordinate phrase
                    if left_phrase and right_phrase:
                        coord_list.append(f"{left_phrase} and {right_phrase}")
                
                # Recursively check for nested NP, VP, PP in subtrees
                sub_np, sub_vp, sub_pp, sub_coord = extract_phrases_from_tree(subtree)
                np_list.extend(sub_np)
                vp_list.extend(sub_vp)
                pp_list.extend(sub_pp)
                coord_list.extend(sub_coord)
    
    return np_list, vp_list, pp_list, coord_list

# Connect to the CoreNLP server
nlp = StanfordCoreNLP('http://localhost', port=9999)

# Sample sentence for testing
sentence = "He called a deep-sea lander overboard and watched as it sank into the cold, dark waters."

# Tokenize and Constituency Parsing
tokens = nlp.word_tokenize(sentence)
constituency_parse = nlp.parse(sentence)

# Print out the constituency parse tree to check its validity
print(f"Constituency Parse Tree for the sentence:")
print(constituency_parse)

# Parse the constituency tree
try:
    parse_tree = nltk.Tree.fromstring(constituency_parse)
except Exception as e:
    print(f"Error in parsing tree: {e}")

# Extract NP, VP, PP, and Coordinated Phrases
np_list, vp_list, pp_list, coord_list = extract_phrases_from_tree(parse_tree)

# Directly print the extracted NP, VP, PP, and Coordinate Phrases
print(f"\nExtracted NP List: {np_list}")
print(f"Extracted VP List: {vp_list}")
print(f"Extracted PP List: {pp_list}")
print(f"Extracted Coordinate Phrases: {coord_list}")

# Calculate mean length and count of NP, VP, PP, and coordinate phrases
mean_np_length = sum(len(n.split()) for n in np_list) / len(np_list) if np_list else 0
mean_vp_length = sum(len(v.split()) for v in vp_list) / len(vp_list) if vp_list else 0
mean_pp_length = sum(len(p.split()) for p in pp_list) / len(pp_list) if pp_list else 0
mean_coord_length = sum(len(c.split()) for c in coord_list) / len(coord_list) if coord_list else 0

# Calculate the incidence (count) of coordinate phrases
coord_count = len(coord_list)

# Print the final summary of the results
print(f"\nSummary:")
print(f"Mean NP Length: {mean_np_length}")
print(f"Mean VP Length: {mean_vp_length}")
print(f"Mean PP Length: {mean_pp_length}")
print(f"Mean Coordinate Length: {mean_coord_length}")
print(f"NP Count: {len(np_list)}")
print(f"VP Count: {len(vp_list)}")
print(f"PP Count: {len(pp_list)}")
print(f"Coordinate Count: {coord_count}")