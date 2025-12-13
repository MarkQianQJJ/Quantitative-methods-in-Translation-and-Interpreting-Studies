from stanfordcorenlp import StanfordCoreNLP
#D:\vscode\stanford-corenlp-4.5.8>java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9999 -timeout 15000
# Connect to the CoreNLP server running on port 9999
nlp = StanfordCoreNLP('http://localhost', port=9999)

sentence = "A colony of red wolves that was reintroduced in North Carolina in 1987 is failing because of poor management and fierce state opposition from game officials and hunters who are killing it, said the five-year review, prepared by the U.S. Fish and Wildlife Service's Southeastern Regional Office and released Tuesday, April 24."

# Tokenization
tokens = nlp.word_tokenize(sentence)
print("Tokens:", tokens)

# Part-of-Speech (POS) Tagging
pos_tags = nlp.pos_tag(sentence)
print("POS Tags:", pos_tags)

# Named Entity Recognition (NER)
ner_tags = nlp.ner(sentence)
print("NER Tags:", ner_tags)

# Constituency Parsing
constituency = nlp.parse(sentence)
print("constituency:", constituency)

# Dependency Parsing
dependencies = nlp.dependency_parse(sentence)
print("Dependencies:", dependencies)

# Close the connection
nlp.close()