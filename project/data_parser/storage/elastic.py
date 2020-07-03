from elasticsearch import Elasticsearch, helpers

class ElasticsearchHelper:
    def __init__(self, es_host, name, analyzer='my_analyzer'):
        self.__es = Elasticsearch(hosts = [es_host])
        self.__name = name
        self.__analyzer = analyzer

    def setIndex(self, name):
        self.__name = name

    def exists(self):
        return self.__es.indices.exists(self.__name)

    def delete(self):
        if self.exists():
            return self.__es.indices.delete(index=self.__name)

    def create(self, delete=False, **args):
        if delete:
            self.delete()
        if not self.exists():
            return self.__es.indices.create(index = self.__name, **args)

    def bulk(self, **args):
        return self.__es.bulk(index = self.__name, **args)

    def streaming_bulk(self, **args):
        return helpers.streaming_bulk(self.__es, index=self.__name, **args)

    def search(self, **args):
        return self.__es.search(index = self.__name, **args)

    def get(self, **args):
        return self.__es.get(index=self.__name, **args)

    def mtermvectors(self, ids, **args):
        i=0
        batch = 200
        while i<len(ids):
            str_ids=','.join(ids[i:i+batch])
            docs = self.__es.mtermvectors(index=self.__name, ids=str_ids, **args)['docs']
            for doc in docs:
                del doc['_index']
                del doc['_type']
                del doc['_id']
                del doc['_version']
                del doc['found']
                del doc['took']
                doc = doc['term_vectors']['text']['terms']
                yield doc
                i += 1
                if i%1000==0:
                    print(i,"docs done")
            

    def tokenize(self, text):
        tokens = self.__es.indices.analyze(index=self.__name, body={
            'analyzer': self.__analyzer,
            'text': text
        })['tokens']
        ret = {}
        for t in tokens:
            ret[t['token']] = text.count(t['token'])
        return ret

    def mget(self, **args):
        ids = args["body"]["ids"]
        if len(ids) < 1000:
            return self.__es.mget(index=self.__name, **args)['docs']
        else:
            res = []
            while len(ids) > 0:
                args["body"]={"ids":ids[:1000]}
                ids = ids[1000:]
                res.extend(self.__es.mget(index=self.__name, **args)['docs'])
            return res

