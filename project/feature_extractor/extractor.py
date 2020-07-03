class Extractor:
    def __init__(self, feature_list):
        self.__feature_list=feature_list
    
    def transform(self, query_doc_map, file):
        for i, docs in enumerate(query_doc_map):
            j = 0
            for id, label in docs:
                row = []
                for feat in self.__feature_list:
                    row.extend(feat.apply(i, id))
                row.append(label)
                file.write(','.join('{:.5f}'.format(r) for r in row)+'\n')
                if j % 10==0:
                    print(j, "Docs done for query", i)
                j += 1
            if i % 1 == 0:
                print(i, "Queries done")
        file.close()
