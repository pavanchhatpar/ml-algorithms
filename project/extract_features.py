from feature_extractor import Vectorizer, Extractor
from data_parser.storage import ElasticsearchHelper
from feature_extractor.features import BM25, JMercer, LM, CosineSimilarity, TfidfFeatures, ESBuiltIn
import numpy as np

qrel_root="/home/pavan/Projects/MS Projects/ML/trec8/qrel.trec8"
qrel = np.genfromtxt(qrel_root,dtype=str, delimiter=" ")
ES_HOST = {
    "host" : "localhost", 
    "port" : 9200
}
TYPE="doc"
index="trec_filtered"
es = ElasticsearchHelper(ES_HOST, index)
vectorizer = Vectorizer(pickled_file='tfidfvectorizer.model')
query_doc_map = []
offset = 401
query_rel_docs = []
for i in range(50):
    query_doc_map.append([])
for i in range(50):
    query_rel_docs.append([])
for qr in qrel:
    query_doc_map[int(qr[0])-offset].append((qr[2], int(qr[3])))
    if qr[3] == '1':
        query_rel_docs[int(qr[0])-offset].append(qr[2])
# total = 0
# for query in query_doc_map:
#     total += len(query)
# print("total", total)
query_file = "/home/pavan/Projects/MS Projects/ML/trec8/topics.401-450_title"
queryFile = np.genfromtxt(query_file, dtype=str, delimiter= ".\t")
queries = []
for query in queryFile:
    queries.append(query[1])
docsIDs = []
for i, qr in enumerate(qrel):
    docsIDs.append(qr[2])
    # if documents.get(qr[2], None) == None:
    #     result = es.get(id=qr[2], doc_type="doc")
    #     documents[qr[2]] = result["_source"]["text"]
    # qrel[i][2] = documents[qr[2]]
    # qrel[i][0] = queryDict[qr[0]]
docsIDs = np.unique(docsIDs)
modC = 0
cqid = {}
docs = es.mtermvectors(ids=docsIDs, doc_type='doc', term_statistics=False, field_statistics=False)
for doc in docs:
    for term, data in doc.items():
        modC += data['term_freq']
        cqid[term] = cqid.get(term, 0) + data['term_freq']

features = [
    # CosineSimilarity(vectorizer, es, docsIDs, queries),
    # LM(vectorizer, es, docsIDs, queries, modC, cqid),
    # JMercer(vectorizer, es, docsIDs, queries, modC, cqid),
    # BM25(vectorizer, es, docsIDs, queries, query_rel_docs),
    # TfidfFeatures(vectorizer, es, docsIDs, queries)
    ESBuiltIn(vectorizer, es, docsIDs, queries)
]

extractor = Extractor(features)

extractor.transform(query_doc_map, open("/home/pavan/Projects/MS Projects/ML/trec8/esbuiltin.csv", "w"))