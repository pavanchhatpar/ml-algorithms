import numpy as np

class TfidfFeatures:
    def __init__(self, vectorizer, es, docIDs, queries):
        self.__docIDs = docIDs
        self.__vectorizer = vectorizer
        self.__es = es
        self.__queries = queries

    def apply(self, queryID, docID):
        tokens=self.__es.tokenize(self.__queries[queryID])
        docTokens=next(self.__es.mtermvectors(ids=[docID], doc_type='doc', field_statistics=False))
        formatted = {}
        idf = 0
        number_of_query_tokens_in_doc = 0
        tfs = []
        for term, freq in tokens.items():
            if term in docTokens:
                number_of_query_tokens_in_doc += 1
                formatted[term] = docTokens[term]
                tfs.append(docTokens[term]['term_freq'])
                idf += np.log(self.__vectorizer.model['N']/(self.__vectorizer.model['doc_freq'][term]+1))
        tfidf = self.__vectorizer.transform([formatted])
        tfs = np.array(tfs)
        try:
            minV = tfidf.data.min()
            maxV = tfidf.data.max()

            return [number_of_query_tokens_in_doc, number_of_query_tokens_in_doc/len(tokens), idf, tfs.sum(), tfs.max(), tfs.min(), tfs.mean(), tfs.var(), tfidf.sum(), maxV, minV, tfidf.mean(), tfidf.toarray().var()]
        except:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
