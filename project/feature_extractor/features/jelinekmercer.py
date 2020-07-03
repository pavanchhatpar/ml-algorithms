import numpy as np

class JMercer:
    def __init__(self, vectorizer, es, docIDs, queries, modC, cqid, mu=0.35):
        self.__mu = mu
        self.__docIDs = docIDs
        self.__vectorizer = vectorizer
        self.__es = es
        self.__queries = queries
        self.__modC = modC
        self.__cqid = cqid

    def apply(self, queryID, docID):
        score = 0
        query_tokens = self.__es.tokenize(self.__queries[queryID])
        doc = next(self.__es.mtermvectors([docID], doc_type='doc', term_statistics=False, field_statistics=False))
        modD = 0
        for term, data in doc.items():
            modD += data['term_freq']
        for token, freq in query_tokens.items():
            t_index = self.__vectorizer.model['term_indices'].get(token, -1)
            if t_index == -1:
                continue
            idf = np.log(self.__vectorizer.model['N']/(self.__vectorizer.model['doc_freq'][token] + 1))
            
            fqid = doc.get(token, {'term_freq':0})['term_freq']
            score += np.log((1-self.__mu)*fqid/modD + self.__mu*self.__cqid.get(token, 0)/self.__modC)
        return [score]


