import numpy as np

class KFold:
    def __init__(self, k, shuffle=True):
        self.__k = k
        self.__shuffle = shuffle

    def split(self, X, y):
        if (len(X) != len(y)):
            raise ValueError("X and y should have same number of data points")
        l = len(X)
        indices = [i for i in range(l)]
        k_indices = []
        if self.__shuffle == None:
            for i in range(self.__k):
                k_indices.append([j for j in range(i,len(indices),self.__k)])
        else:
            if self.__shuffle:
                np.random.shuffle(indices)
            window_size = l // self.__k
            
            for i in range(self.__k-1):
                k_indices.append([indices[j] for j in range(window_size*i, window_size*(i+1))])
            k_indices.append([indices[j] for j in range(window_size*(self.__k-1), l)])
        splits = []
        for i in range(self.__k):
            splits.append((np.concatenate([x for j, x in enumerate(k_indices) if j != i]).ravel(), np.array(k_indices[i])))
        return splits