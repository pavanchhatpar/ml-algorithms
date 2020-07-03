from pymongo import MongoClient, ASCENDING

class MongoDBHelper:
    def __init__(self, dbName):
        self.__dbName = dbName
        self.__client = MongoClient()
        self.__db = self.__client[self.__dbName]

    def setCollectionName(self, collectionName, uniqueIndices=None):
        self.__collectionName = collectionName
        self.__collection = self.__db[self.__collectionName]
        if uniqueIndices != None:
            uids = []
            for index in uniqueIndices:
                uids.append((index, ASCENDING)) 
            self.__collection.create_index(uids, unique=True)
    
    def getCollectionName(self):
        return self.__collectionName

    def close(self):
        self.__client.close()

    def insert(self, data):
        return self.__collection.insert_one(data).inserted_id

    def insert_many(self, data):
        return self.__collection.insert_many(data).inserted_ids

    def find(self, query={}, projection={}):
        return self.__collection.find(query, projection)
