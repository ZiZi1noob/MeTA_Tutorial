#using conda powershell terminal
#"python search_eval.py config.toml" -- 'search-eval.py' is search file; 'config.toml' is parameter
#1. setting '起始位置' in powershell because 'target documents or script' should be in directory of powershell
#2. add 'target document path' in '起始位置' 
#3. 'conda activate python3.7' select python version
import math
import sys
import time

import metapy
import pytoml
from scipy import stats

class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, some_param): #some_param is for setting c in the formula
        self.param = some_param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd): #'self' represents the instance of the class and is always pointing to the current object
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        tfn = sd.doc_term_count * math.log(1 + (sd.avg_dl / sd.doc_size), 2)
        # set c = self.param;  # when c = 6 max
        score = sd.query_term_weight *(tfn / (tfn + self.param)) * math.log((sd.num_docs + 1)/ (sd.corpus_term_count + 0.5), 2)
        return score
def write_file(file_name, file_contents):
    with open (file_name, 'w') as f:
        for line in file_contents:
            f.write(str(line))
            f.write('\n')

def load_ranker_DirPri(cfg_file):

    return metapy.index.DirichletPrior(mu = 6) # DirichletPrior Ranker
    
def load_ranker_InL2(cfg_file):

    return InL2Ranker(some_param=6) #Inl2 Ranker


#main function
if __name__ == '__main__':  #https://stackoverflow.com/questions/419163/what-does-if-name-main-do
    if len(sys.argv) != 2: #count number of arguments   #sys.argv: ['search_eval.py', 'config.toml']
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)
    
    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ranker_DirPri = load_ranker_DirPri(cfg)
    ranker_InL2 = load_ranker_InL2(cfg)
    ev = metapy.index.IREval(cfg) #evaluation

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin) #cfg_d = {'prefix': '.', 'stop-words': 'stopwords.txt', 'dataset': 'cranfield', 'corpus': 'line.toml', 'index': 'idx', 'query-judgements': 'cranfield-qrels.txt', 'analyzers': [{'method': 'ngram-word', 'ngram': 1, 'filter': 'default-unigram-chain'}], 'query-runner': {'query-path': 'cranfield-queries.txt', 'query-id-start': 1}}

    query_cfg = cfg_d['query-runner'] #query_cfg = {'query-path': 'cranfield-queries.txt', 'query-id-start': 1}
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt') # query_path: cranfield-queries.txt
    query_start = query_cfg.get('query-id-start', 0)
    
    query = metapy.index.Document()
    print('Running queries')

    inl2_AP_file = 'inl2.AP_file.txt'
    DirichletPrior_AP_file = 'DirichletPrior_AP_file.txt'
    ttest_file = 'significance.txt'
    
    InL2_box = []
    DirichletPrior_AP_box = []
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file): #query_num is line numbers; line is contents of each line
            query.content(line.strip())
            results_DirPri = ranker_DirPri.score(idx, query, top_k)  #top_k scores list
            results_InL2 = ranker_InL2.score(idx, query, top_k)  #top_k scores list
            #print('{}th results is {}'.format(query_num, results))
            avg_p_DP = ev.avg_p(results_DirPri, query_start + query_num, top_k) #top_k's average precision; # 'query_start+query_num'
            avg_p_IL2 = ev.avg_p(results_InL2, query_start + query_num, top_k)
            #print('{}th avg_p is {}'.format(query_num, avg_p))
            #print(query_start + query_num)
            DirichletPrior_AP_box.append(avg_p_DP)
            InL2_box.append(avg_p_IL2)
            #print("Query {} average precision: {}".format(query_num + 1, avg_p))
    #print("Mean average precision: {}".format(ev.map()))
    write_file(DirichletPrior_AP_file, DirichletPrior_AP_box) # DirichletPrior Ranker
    write_file(inl2_AP_file, InL2_box)
    print('DirichletPrior_AP_file.txt and inl2.AP_file.txt are ready')
    
    ttest_result = stats.ttest_rel(InL2_box, DirichletPrior_AP_box)[1]
    with open (ttest_file, 'w') as f:
        f.write(str(ttest_result))
        f.write('\n')
    print('T-test value done and the value is:', ttest_result)
    # print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))

    a = cfg_d.get('query-judgements')
    with open(a) as query_file:
        #for query_num, content in enumerate(query_file):
           # print(content)
        None
