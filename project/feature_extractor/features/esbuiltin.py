import numpy as np

class ESBuiltIn:
    def __init__(self, vectorizer, es, docIDs, queries):
        self.__vectorizer = vectorizer
        self.__es = es
        self.__docIDs = docIDs #probably not needed
        self.__queries = queries
        #TODO: run es.search and get doc scores, if no doc score present then 0
        self.__scores = []
        for query in queries:
            results = self.__es.search(doc_type='doc', body={"query": {"match":{"text":query}}})['hits']['hits']
            docs = {}
            for res in results:
                docs[res["_id"]] = res["_score"]
            self.__scores.append(docs)
    
    def apply(self, queryID, docID):
        return [self.__scores[queryID].get(docID, 0)]