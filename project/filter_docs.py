import numpy as np
from data_parser.storage import ElasticsearchHelper
qrel_root="/home/pavan/Projects/MS Projects/ML/trec8/qrel.trec8"
qrel = np.genfromtxt(qrel_root,dtype=str, delimiter=" ")

# query_file = "/home/pavan/Projects/MS Projects/ML/trec8/topics.401-450_title"
# queries = np.genfromtxt(query_file, dtype=str, delimiter= ".\t")
# queryDict = {}
# for query in queries:
#     queryDict[query[0]] = query[1]
# documents = {}
ES_HOST = {
    "host" : "localhost", 
    "port" : 9200
}
TYPE="doc"
index="trec"
es = ElasticsearchHelper(ES_HOST, index)
docsIDs = []
for i, qr in enumerate(qrel):
    docsIDs.append(qr[2])
    # if documents.get(qr[2], None) == None:
    #     result = es.get(id=qr[2], doc_type="doc")
    #     documents[qr[2]] = result["_source"]["text"]
    # qrel[i][2] = documents[qr[2]]
    # qrel[i][0] = queryDict[qr[0]]
docsIDs = np.unique(docsIDs)
def parse(docsIDs):
    print("Fetching documents from index")
    res=es.mget(doc_type="doc", body={"ids":docsIDs.tolist()})
    docTexts = [r['_source']['text'] for r in res]
    for i, doc in enumerate(docTexts):
        yield {
            "_index": "trec_filtered",
            "_op_type": "index",
            "_type": TYPE,
            "_id": docsIDs[i],
            "text": doc
        }
body={
    "mappings": {
        "doc": {
        "properties": {
            "text": {
            "type":        "text",
            "term_vector": "yes",
            "analyzer":    "my_analyzer"
            }
        }
        }
    },
    "settings": {
        "analysis": {
        "filter": {
            "english_stop": {
            "type":       "stop",
            "stopwords":  "_english_" 
            }
        },
        "analyzer": {
            "my_analyzer": {
            "tokenizer": "my_tokenizer",
            "filter": [
                "lowercase",
                "english_stop"
            ]
            }
        },
        "tokenizer": {
            "my_tokenizer": {
            "type": "ngram",
            "min_gram": 3,
            "max_gram": 3,
            "token_chars": [
                "letter",
                "digit"
            ]
            }
        }
        }
    }
}
es1 = ElasticsearchHelper(ES_HOST, "trec_filtered")
es1.create(delete=True, body=body)
for ok, result in es1.streaming_bulk(actions=parse(docsIDs), doc_type=TYPE, chunk_size=1500):
    action, result = result.popitem()
    doc_id = '/%s/%s/%s' % ("trec_filtered", TYPE, result['_id'])
    # process the information from ES whether the document has been
    # successfully indexed
    if not ok:
        print('Failed to %s document %s: %r' % (action, doc_id, result))
    else:
        print(doc_id)
# print("Extracting features from text")
# vectorizer = Vectorizer(argsCount={"stop_words":'english', "ngram_range":(1,2), "min_df":0.001})
# vectorizer.fit(docTexts)
# vectorizer.export('tfidfvectorizer.model')