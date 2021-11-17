
# Setup
We'll use [metapy](https://github.com/meta-toolkit/metapy)---Python bindings for MeTA. , use the following commands to get started. 

**Please note that students have had issues using metapy with specific Python versions in the past (e.g. Python 3.7 on mac). To avoid issues, please use Python 2.7 or 3.5. Your code will be tested using Python 3.5** 

```bash
# Ensure your pip is up to date
pip install --upgrade pip

# install metapy!
pip install metapy pytoml
```


## Start
Let's start by importing metapy. Open a terminal and type
```bash
python
``` 
to get started

```python
#import the MeTA python bindings
import metapy
#If you'd like, you can tell MeTA to log to stderr so you can get progress output when running long-running function calls.
metapy.log_to_stderr()
```


Now, let's create a document with some content to experiment on
```python
doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")
```

## Tokenization
MeTA provides a stream-based interface for performing document tokenization.
Each stream starts off with a Tokenizer object, and in most cases you should use the Unicode standard aware ICUTokenizer.

```python
tok = metapy.analyzers.ICUTokenizer()
```

Tokenizers operate on raw text and provide an Iterable that spits out the individual text tokens.
Let's try running just the ICUTokenizer to see what it does.

```python
tok.set_content(doc.content()) # this could be any string
tokens = [token for token in tok]
print(tokens)
```

One thing that you likely immediately notice is the insertion of these pseudo-XML looking tags.
These are called “sentence boundary tags”.
As a side-effect, a default-construted ICUTokenizer discovers the sentences in a document by delimiting them with the sentence boundary tags.
Let's try tokenizing a multi-sentence document to see what that looks like.

```python
doc.content("I said that I can't believe that it only costs $19.95! I could only find it for more than $30 before.")
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Most of the information retrieval techniques you have likely been learning about in this class don't need to concern themselves with finding the boundaries between separate sentences in a document, but later today we'll explore a scenario where this might matter more.
Let's pass a flag to the ICUTokenizer constructor to disable sentence boundary tags for now.

```python
tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

As mentioned earlier, MeTA treats tokenization as a streaming process, and that it starts with a tokenizer.
It is often beneficial to modify the raw underlying tokens of a document, and thus change its representation.
The “intermediate” steps in the tokenization stream are represented with objects called Filters.
Each filter consumes the content of a previous filter (or a tokenizer) and modifies the tokens coming out of the stream in some way.
Let's start by using a simple filter that can help eliminate a lot of noise that we might encounter when tokenizing web documents: a LengthFilter.

```python
tok = metapy.analyzers.LengthFilter(tok, min=2, max=30)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Here, we can see that the LengthFilter is consuming our original ICUTokenizer.
It modifies the token stream by only emitting tokens that are of a minimum length of 2 and a maximum length of 30.
This can get rid of a lot of punctuation tokens, but also excessively long tokens such as URLs.

## Stopword removal and stemming

Another common trick is to remove stopwords. In MeTA, this is done using a ListFilter.

```bash
wget -nc https://raw.githubusercontent.com/meta-toolkit/meta/master/data/lemur-stopwords.txt
```

Note: wget is a command to download files from links. Another simpler option is to open a web browser, type the link on the address bar and download the file manually

```python
tok = metapy.analyzers.ListFilter(tok, "lemur-stopwords.txt", metapy.analyzers.ListFilter.Type.Reject)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Here we've downloaded a common list of stopwords and created a ListFilter to reject any tokens that occur in that list of words.
You can see how much of a difference removing stopwords can make on the size of a document's token stream!

Another common filter that people use is called a stemmer, or lemmatizer.
This kind of filter tries to modify individual tokens in such a way that different inflected forms of a word all reduce to the same representation.
This lets you, for example, find documents about a “run” when you search “running” or “runs”.
A common stemmer is the Porter2 Stemmer, which MeTA has an implementation of.
Let's try it!

```python
tok = metapy.analyzers.Porter2Filter(tok)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

## N-grams

Finally, after you've got the token stream configured the way you'd like, it's time to analyze the document by consuming each token from its token stream and performing some actions based on these tokens.
In the simplest case, our action can simply be counting how many times these tokens occur.
For clarity, let's switch back to a simpler token stream first.
We will write a token stream that tokenizes with ICUTokenizer, and then lowercases each token.

```python
tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
tok = metapy.analyzers.LowercaseFilter(tok)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Now, let's count how often each individual token appears in the stream.
This representation is called “bag of words” representation or “unigram word counts”.
In MeTA, classes that consume a token stream and emit a document representation are called Analyzers.

```python
ana = metapy.analyzers.NGramWordAnalyzer(1, tok)
print(doc.content())
unigrams = ana.analyze(doc)
print(unigrams)
```

If you noticed the name of the analyzer, you might have realized that you can count not just individual tokens, but groups of them.
“Unigram” means “1-gram”, and we count individual tokens. “Bigram” means “2-gram”, and we count adjacent tokens together as a group.
Let's try that now.

```python
ana = metapy.analyzers.NGramWordAnalyzer(2, tok)
bigrams = ana.analyze(doc)
print(bigrams)
```

Now the individual “tokens” we're counting are pairs of tokens.
Sometimes looking at n-grams of characters is useful.

```python
tok = metapy.analyzers.CharacterTokenizer()
ana = metapy.analyzers.NGramWordAnalyzer(4, tok)
fourchar_ngrams = ana.analyze(doc)
print(fourchar_ngrams)
```

# POS tagging

Now, let's explore something a little bit different.
MeTA also has a natural language processing (NLP) component, which currently supports two major NLP tasks: part-of-speech tagging and syntactic parsing.
POS tagging is a task in NLP that involves identifying a type for each word in a sentence.
For example, POS tagging can be used to identify all of the nouns in a sentence, or all of the verbs, or adjectives, or…
This is useful as first step towards developing an understanding of the meaning of a particular sentence.
MeTA places its POS tagging component in its “sequences” library.
Let's play with some sequences first to get an idea of how they work.
We'll start of by creating a sequence.

```python
seq = metapy.sequence.Sequence()
```

Now, we can add individual words to this sequence.
Sequences consist of a list of Observations, which are essentially (word, tag) pairs.
If we don't yet know the tags for a Sequence, we can just add individual words and leave the tags unset.
Words are called “symbols” in the library terminology.

```python
for word in ["The", "dog", "ran", "across", "the", "park", "."]:
    seq.add_symbol(word)

print(seq)
```

The printed form of the sequence shows that we do not yet know the tags for each word.
Let's fill them in by using a pre-trained POS-tagger model that's distributed with MeTA.

```bash
wget -nc https://github.com/meta-toolkit/meta/releases/download/v3.0.1/greedy-perceptron-tagger.tar.gz
tar xvf greedy-perceptron-tagger.tar.gz
```

```python
tagger = metapy.sequence.PerceptronTagger("perceptron-tagger/")
tagger.tag(seq)
print(seq)
```

Each tag indicates the type of a word, and this particular tagger was trained to output the tags present in the Penn Treebank tagset.
But what if we want to POS-tag a document?

```python
doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")
tok = metapy.analyzers.ICUTokenizer() # keep sentence boundaries!
tok = metapy.analyzers.PennTreebankNormalizer(tok)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Now, we will write a function that can take a token stream that contains sentence boundary tags and returns a list of Sequence objects.
We will not include the sentence boundary tags in the actual Sequence objects.

```python
def extract_sequences(tok):
    sequences = []
    for token in tok:
        if token == '<s>':
            sequences.append(metapy.sequence.Sequence())
        elif token != '</s>':
            sequences[-1].add_symbol(token)
    return sequences

doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")
tok.set_content(doc.content())
for seq in extract_sequences(tok):
    tagger.tag(seq)
    print(seq)
```

## Config.toml file: setting up a pipeline

In practice, it is often beneficial to combine multiple feature sets together.
We can do this with a MultiAnalyzer. Let's combine unigram words, bigram POS tags, and rewrite rules for our document feature representation.
We can certainly do this programmatically, but doing so can become tedious quite quickly.
Instead, let's use MeTA's configuration file format to specify our analyzer, which we can then load in one line of code.
MeTA uses TOML configuration files for all of its configuration. If you haven't heard of TOML before, don't panic! It's a very simple, readable format.
Open a text editor and copy the text below, but be careful not to modify the contents. Save it as `config.toml` .

```
#Add this as a config.toml file to your project directory
stop-words = "lemur-stopwords.txt"

[[analyzers]]
method = "ngram-word"
ngram = 1
filter = "default-unigram-chain"

[[analyzers]]
method = "ngram-pos"
ngram = 2
filter = [{type = "icu-tokenizer"}, {type = "ptb-normalizer"}]
crf-prefix = "crf"

[[analyzers]]
method = "tree"
filter = [{type = "icu-tokenizer"}, {type = "ptb-normalizer"}]
features = ["subtree"]
tagger = "perceptron-tagger/"
parser = "parser/"
```

Each [[analyzers]] block defines another analyzer to combine for our feature representation.
Since “ngram-word” is such a common analyzer, we have defined some default filter chains that can be used with shortcuts.
“default-unigram-chain” is a filter chain suitable for unigram words; “default-chain” is a filter chain suitable for bigram words and above.

To run this example, we will need to download some additional MeTA resources:

```bash
wget -nc https://github.com/meta-toolkit/meta/releases/download/v3.0.2/crf.tar.gz
tar xvf crf.tar.gz
```

```bash
wget -nc https://github.com/meta-toolkit/meta/releases/download/v3.0.2/greedy-constituency-parser.tar.gz
tar xvf greedy-constituency-parser.tar.gz
```

We can now load an analyzer from this configuration file:

```python
ana = metapy.analyzers.load('config.toml')
doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")
print(ana.analyze(doc))
```

## Trying it out on your own!
Try it in Example
# CS410 MP2---Search Engines

In this 4-part MP, you will get familiar with building and evaluating Search Engines.

## Part 2

### Due: Sept 19, 2021 

In this part, you will use the MeTA toolkit to do the following:
- create a search engine over a dataset
- investigate the effect of parameter values for a standard retrieval function
- write the InL2 retrieval function
- investigate the effect of the parameter value for InL2


Also, you are free to edit all files **except** 
- livedatalab_config.json


## Setup
We'll use [metapy](https://github.com/meta-toolkit/metapy)---Python bindings for MeTA. 
If you have not installed metapy so far, use the following commands to get started.

```bash
# Ensure your pip is up to date
pip install --upgrade pip

# install metapy!
pip install metapy pytoml
```

Read the [C++ Search Tutorial](https://meta-toolkit.org/search-tutorial.html). Read *Initially setting up the config file and Relevance judgements*.
Read the [python Search Tutorial](https://github.com/meta-toolkit/metapy/blob/master/tutorials/2-search-and-ir-eval.ipynb)

We have provided the following files:
- Cranfield dataset in MeTA format.
- cranfield-queries.txt: Queries one per line
- cranfield-qrels.txt: Relevance judgements for the queries
- stopwords.txt: A file containing stopwords that will not be indexed.
- config.toml: A config file with paths set to all the above files, including index and ranker settings.

## Indexing the data
To index the data using metapy, use the following .
```python
import metapy
idx = metapy.index.make_inverted_index('config.toml')
```

## Search the index
You can examine the data inside the cranfield directory to get a sense about the dataset and the queries.

To examine the index we built from the previous section. You can use metapy's functions.

```python
# Examine number of documents
idx.num_docs()
# Number of unique terms in the dataset
idx.unique_terms()
# The average document length
idx.avg_doc_length()
# The total number of terms
idx.total_corpus_terms()
```

Here is a list of all the rankers in MeTA.Viewing the class comment in the header files shows the optional parameters you can set in the config file:

- [Okapi BM25](https://github.com/meta-toolkit/meta/blob/master/include/meta/index/ranker/okapi_bm25.h), method = "**bm25**" 
- [Pivoted Length Normalization](https://github.com/meta-toolkit/meta/blob/master/include/meta/index/ranker/pivoted_length.h), method = "**pivoted-length**"
- [Absolute Discount Smoothing](https://github.com/meta-toolkit/meta/blob/master/include/meta/index/ranker/absolute_discount.h), method = "**absolute-discount**"
- [Jelinek-Mercer Smoothing](https://github.com/meta-toolkit/meta/blob/master/include/meta/index/ranker/jelinek_mercer.h), method = "**jelinek-mercer**"
- [Dirichlet Prior Smoothing](https://github.com/meta-toolkit/meta/blob/master/include/meta/index/ranker/dirichlet_prior.h), method = "**dirichlet-prior**"

In metapy, the rankers can be called as:

```python
metapy.index.OkapiBM25(k1, b, k3) where k1, b, k3 are function arguments, e.g. ranker = metapy.index.OkapiBM25(k1=1.2,b=0.75,k3=500)
metapy.index.PivotedLength(s) 
metapy.index.AbsoluteDiscount(delta)
metapy.index.JelinekMercer(lambda)
metapy.index.DirichletPrior(mu)
```

## Varying a parameter
Choose one of the above retrieval functions and one of its parameters (don’t choose BM25 + k3, it’s not interesting). For example, you could choose Dirichlet Prior and mu.

Change the **ranker** to your method and parameters. In the example, it is set to **bm25**. Use at least 10 different values for the parameter you chose; try to choose the values such that you can find a maximum MAP.

Here's a tutorial on how to do an evaluation of your parameter setting (this code is included in *search_eval.py*):


```python
# Build the query object and initialize a ranker
query = metapy.index.Document()
ranker = metapy.index.OkapiBM25(k1=1.2,b=0.75,k3=500)
# To do an IR evaluation, we need to use the queries file and relevance judgements.
ev = metapy.index.IREval('config.toml')
# Load the query_start from config.toml or default to zero if not found
with open('config.toml', 'r') as fin:
        cfg_d = pytoml.load(fin)
query_cfg = cfg_d['query-runner']
query_start = query_cfg.get('query-id-start', 0)
# We will loop over the queries file and add each result to the IREval object ev.
num_results = 10
with open('cranfield-queries.txt') as query_file:
    for query_num, line in enumerate(query_file):
        query.content(line.strip())
        results = ranker.score(idx, query, num_results)                            
        avg_p = ev.avg_p(results, query_start + query_num, num_results)
        print("Query {} average precision: {}".format(query_num + 1, avg_p))
ev.map()
```

## Writing InL2

You will now implement a retrieval function called InL2. It is described in [this](http://dl.acm.org/citation.cfm?id=582416) paper: 
For this assignment, we will only concern ourselves with writing the function and not worry about its derivation. 
InL2 is formulated as 

![image](https://drive.google.com/uc?export=view&id=1_Q2CTMe6o2RP9PGf8HPsggai9LVyVmEU) 

Please use this link if the image does not display: https://drive.google.com/uc?export=view&id=1_Q2CTMe6o2RP9PGf8HPsggai9LVyVmEU


, where

![image](https://drive.google.com/uc?export=view&id=1gcbywLx0ZEU3eqxlDtLk6o4Yxd788IiK)

Please use this link if the image does not display: https://drive.google.com/uc?export=view&id=1gcbywLx0ZEU3eqxlDtLk6o4Yxd788IiK

It uses the following variables:

- <em> Q,D,t </em> : the current query, document, and term
- <em> N </em> : the total number of documents in the corpus C
- <em> avgdl </em> : the average document length
- <em> c > 0 </em> : is a parameter

Determine if this function captures the TF, IDF, and document length normalization properties. Where (if anywhere) are they represented in the formula? You don’t need to submit your answers.

To implement InL2, define your own ranking function in Python, as shown below. 
You do not need to create a new file, the template is included in *search_eval.py*  You will need to modify the function **score_one**. 
Do not forget to call the InL2 ranker by editing the return statement of *load_ranker* function inside search_eval.py.

The parameter to the function is a score_data sd object. See the object [here](https://github.com/meta-toolkit/meta/blob/master/include/meta/index/score_data.h).

As you can see, the sd variable contains all the information you need to write the scoring function. The function you’re writing represents one term in the large InL2 sum.

```python
class InL2Ranker(metapy.index.RankingFunction):                                            
    """                                                                          
    Create a new ranking function in Python that can be used in MeTA.             
    """                                                                          
    def __init__(self, some_param=1.0):                                             
        self.param = some_param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()                                        
                                                                                 
    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        return (self.param + sd.doc_term_count) / (self.param * sd.doc_unique_terms + sd.doc_size)
```


## Varying InL2’s parameter
Perform the same parameter analysis with InL2’s <em> c </em> parameter. 

## Statistical significance testing

Modifying the code in "Varying a parameter" section, you can create a file with average precision data. 

Use BM25 as a ranker and create a file called bm25.avg_p.txt. 

Then use your ranker InL2 and create a file called inl2.avg_p.txt. 

Each of these files is simply a list of the APs from the queries.

We want to test whether the difference between your two optimized retrieval functions is statistically significant.

If you’re using R, you can simply do

```R
bm25 = read.table('bm25.avg_p.txt')$V1
inl2 = read.table('inl2.avg_p.txt')$V1
t.test(bm25, inl2, paired=T)
```

You don’t have to use R; you can even write a script to calculate the answer yourself.

In Python, you can use [this function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)

The output of the significance test will give you a p-value. If the p-value is less than 0.05 (our chosen significance level), then we will say that there is a significant difference between the two average precision lists. That means that there is less than a 5% chance that the difference in the mean of the AP scores is due to random fluctuation.

Write the p-value in a file called **significance.txt**. 
***Do not include anything else in the file, just this number!***

## Grading

Your grade will be based on:
- implementing the InL2 parameter correctly -- 0.7 (70) points
- uploading significance.txt with the p-value. -- 0.3 (30) points
