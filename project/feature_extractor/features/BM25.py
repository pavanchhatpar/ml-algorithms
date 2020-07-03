import numpy as np

class BM25:
    def __init__(self, vectorizer, es, docIDs, queries, qrel, k1=1.2, b=0.75, k2=500):
        self.__k1 = k1
        self.__k2 = k2
        self.__b = b
        sum = 0
        for term, freq in vectorizer.model['doc_freq'].items():
            sum += freq
        self.__avdl = sum/vectorizer.model['N']
        self.__docIDs = docIDs
        self.__queries = queries
        self.__qrel = qrel
        self.__vectorizer = vectorizer
        self.__es = es

    def apply(self, queryID, docID):
        score = 0
        # ind = self.__docIDs.index(docID)
        doc = self.__vectorizer.transform(self.__es.mtermvectors([docID], doc_type='doc', term_statistics=False, field_statistics=False),1)
        dl = doc.nnz
        N = self.__vectorizer.model['N']
        R = len(self.__qrel[queryID])
        K = self.__k1*((1-self.__b)+self.__b*dl/self.__avdl)
        query_tokens = self.__es.tokenize(self.__queries[queryID])
        for token, qfi in query_tokens.items():
            ni = self.__vectorizer.model['doc_freq'][token]
            ri = 0
            fi = doc[0, self.__vectorizer.model['term_indices'][token]]
            
            for did in self.__qrel[queryID]:
                # indi = self.__docIDs.index(d)
                d = self.__vectorizer.transform(self.__es.mtermvectors([did], doc_type='doc', term_statistics=False, field_statistics=False),1)
                if d[0, self.__vectorizer.model['term_indices'][token]] != 0:
                    ri += 1
                del d
            score += np.log(((ri+0.5)/(R-ri+0.5))/((ni-ri+0.5)/(N-ni-R+ri+0.5)))
            score += np.log((self.__k1+1)*fi/(K+fi))
            score += np.log((self.__k2+1)*qfi/(self.__k2+qfi))
        return [score]
