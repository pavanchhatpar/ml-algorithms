from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pickle
import numpy as np
from scipy.sparse import csr_matrix

class Vectorizer:
    def __init__(self, pickled_file=None):
        if pickled_file == None:
            self.model = {
                'doc_freq': {},
                'N': 0,
                'term_indices':{}
            }
        else:
            f = open(pickled_file, 'rb')
            self.model = pickle.load(f)
            f.close()
    
    def export(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump(self.model, f)
        f.close()
    
    def fit(self, documents):
        for doc in documents:
            docFreq = self.model['doc_freq']
            # print(len(doc))
            # print("full", len(docFreq))
            N = self.model['N'] + 1
            for term in doc.keys():
                docFreq[term] = docFreq.get(term, 0) + 1
            self.model['doc_freq'] = docFreq
            self.model['N'] = N
            termIndex = self.model['term_indices']
            i = len(termIndex)
            for term in docFreq.keys():
                if termIndex.get(term) == None:
                    termIndex[term] = i
                    i += 1
            self.model['term_indices'] = termIndex

    def transform(self, documents, N=None):
        if N == None:
            ret = csr_matrix((len(documents), len(self.model['doc_freq'])), dtype=np.float32)
        else:
            ret = csr_matrix((N, len(self.model['doc_freq'])), dtype=np.float32)
        idf = {}
        for term, freq in self.model['doc_freq'].items():
            idf[term] = np.log(self.model['N']/(1+freq))
        for i, doc in enumerate(documents):
            for term, data in doc.items():
                idfT = idf.get(term, 0)
                tfidf = data['term_freq']*idfT
                if tfidf != 0:
                    t_index = self.model['term_indices'].get(term, -1)
                    if t_index != -1:
                        ret[i, t_index] = tfidf
        return ret

    def get_feature_names(self):
        ret = np.array(len(self.model['term_indices']), dtype=str)
        for term, i in self.model['term_indices'].items():
            ret[i] = term
        return ret