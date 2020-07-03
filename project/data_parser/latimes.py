
import os
import re

from storage import MongoDBHelper, ElasticsearchHelper
from bs4 import BeautifulSoup as bsoup
data_root = "../../../../Projects/MS Projects/ML/trec8/vol5/latimes"

filenames=[]
for (dirpath, dirnames, fnames) in os.walk(data_root):
    filenames.extend(fnames)

TYPE='doc'
# data = []

ES_HOST = {
    "host" : "localhost", 
    "port" : 9200
}
request_body = {
    "settings" : {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}
index="trec"
es = ElasticsearchHelper(ES_HOST, index)
resCreate = es.create(body=request_body)
def parse_docs():
    for file in filenames:
        
        if not file.startswith('la'):
            continue
        print(file)
        try:
            soup = bsoup(open(os.path.join(data_root,file)), 'html.parser')
        except UnicodeDecodeError:
            soup = bsoup(open(os.path.join(data_root,file), encoding='ISO-8859-15'), 'html.parser')
        docs = soup.find_all('doc')
        for doc in docs:
            # op_dict = {
            #     "index": {
            #         "_index": "trec", 
            #         "_type": TYPE, 
            #         "_id": doc.docno.get_text().strip()
            #     }
            # }
            # data_dict = {
            #     "text": doc.find('text').get_text().strip()
            # }
            # data.append(op_dict)
            # data.append(data_dict)

            try:
                id = doc.docno.extract().get_text().strip()
                try:
                    doc.docid.extract()
                except AttributeError:
                    pass
                try:
                    doc.length.extract()
                except AttributeError:
                    pass
                yield {
                    "_index": index,
                    "_op_type": "index",
                    "_type": TYPE,
                    "_id": id,
                    "text": re.sub('[\s\n]+', ' ', doc.get_text().strip())
                }
            except AttributeError:
                continue
            # data.append({"_id": doc.docno.get_text().strip(), "text": doc.find('text').get_text().strip()})

# db = MongoDBHelper("ML")
# db.setCollectionName("documents")
# db.insert_many(data)


for ok, result in es.streaming_bulk(actions=parse_docs(), doc_type=TYPE, chunk_size=1500):
    action, result = result.popitem()
    doc_id = '/%s/%s/%s' % (index, TYPE, result['_id'])
    # process the information from ES whether the document has been
    # successfully indexed
    if not ok:
        print('Failed to %s document %s: %r' % (action, doc_id, result))
    else:
        print(doc_id)

