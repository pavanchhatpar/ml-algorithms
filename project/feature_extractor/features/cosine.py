import numpy as np

class CosineSimilarity:
    def __init__(self, vectorizer, es, docIDs, queries):
        self.__docIDs = docIDs
        self.__vectorizer = vectorizer
        self.__es = es
        self.__queries = queries

    def apply(self, queryID, docID):
        query_tokens = self.__es.tokenize(self.__queries[queryID])
        formatted = {}
        for term, freq in query_tokens.items():
            formatted[term] = {"term_freq": freq}
        del query_tokens
        return [np.dot(
            self.__vectorizer.transform(self.__es.mtermvectors([docID], doc_type='doc', term_statistics=False, field_statistics=False), 1),
            self.__vectorizer.transform([formatted]).T
            )[0,0]]