from feature_extractor import Vectorizer
from data_parser.storage import ElasticsearchHelper
import numpy as np
qrel_root="/home/pavan/Projects/MS Projects/ML/trec8/qrel.trec8"
qrel = np.genfromtxt(qrel_root,dtype=str, delimiter=" ")
docsIDs = []
for i, qr in enumerate(qrel):
    docsIDs.append(qr[2])
    # if documents.get(qr[2], None) == None:
    #     result = es.get(id=qr[2], doc_type="doc")
    #     documents[qr[2]] = result["_source"]["text"]
    # qrel[i][2] = documents[qr[2]]
    # qrel[i][0] = queryDict[qr[0]]
docsIDs = np.unique(docsIDs)
ES_HOST = {
    "host" : "localhost", 
    "port" : 9200
}
TYPE="doc"
index="trec_filtered"
es = ElasticsearchHelper(ES_HOST, index)
docs = es.mtermvectors(ids=docsIDs, doc_type=TYPE, term_statistics=False, field_statistics=False)
# for i, doc in enumerate(docs):
#     del doc['_index']
#     del doc['_type']
#     del doc['_id']
#     del doc['_version']
#     del doc['found']
#     del doc['took']
#     docs[i] = doc['term_vectors']['text']['terms']
vectorizer = Vectorizer()
vectorizer.fit(docs)
vectorizer.export('tfidfvectorizer.model')