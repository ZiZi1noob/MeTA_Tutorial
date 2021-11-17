import metapy
'''
Use a function that tokenizes with ICUTokenizer (without the end/start tags, i.e. use the argument "suppress_tags=True"),
lowercases, removes words with less than 2 and more than 5 characters, performs stemming and produces trigrams for an input sentence.
Once you edit the example.py to fill in the function, you can check whether your submission passed the tests.
'''


def tokens_lowercase(doc):
    #Write a token stream that tokenizes with ICUTokenizer (use the argument "suppress_tags=True"), 
    #lowercases, removes words with less than 2 and more than 5  characters
    #performs stemming and creates trigrams (name the final call to ana.analyze as "trigrams")
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.LengthFilter(tok, min=2, max=5)
    tok = metapy.analyzers.Porter2Filter(tok)
    ana = metapy.analyzers.NGramWordAnalyzer(3, tok)
    trigrams = ana.analyze(doc)
    #leave the rest of the code as is
    tok.set_content(doc.content())
    tokens, counts = [], []
    for token, count in trigrams.items():
        counts.append(count)
        tokens.append(token)
    return tokens
    
if __name__ == '__main__':
    doc = metapy.index.Document()
    doc.content("I said that I can't believe that it only costs $19.95! I could only find it for more than $30 before.")
    print(doc.content()) #you can access the document string with .content()
    tokens = tokens_lowercase(doc)
    print(tokens)
