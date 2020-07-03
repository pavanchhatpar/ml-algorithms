import numpy as np
path = "../../../Projects/MS Projects/ML/trec8/data.csv"
path2 = "../../../Projects/MS Projects/ML/trec8/tfidf_feats.csv"
path3 = "../../../Projects/MS Projects/ML/trec8/esbuiltin.csv"
qrel_root = "../../../Projects/MS Projects/ML/trec8/qrel.trec8"
data = np.genfromtxt(path, delimiter=",")
data2 = np.genfromtxt(path2, delimiter=",")
data3 = np.genfromtxt(path3, delimiter=",")
qrel = np.genfromtxt(qrel_root,dtype=str, delimiter=" ")
X = data[:,:-7]
X = np.concatenate((X, data2[:,:-1]), axis=1)
X = np.concatenate((X, data3[:,:-1]), axis=1)
# X = MinMaxScaler().fit_transform(X)
y = data[:,-1].astype(int)
query_IDs = qrel[:,0].astype(int)
querys = np.arange(401,451)
output = open("../../../Projects/MS Projects/ML/trec8/ranklibdata.txt", "w")
for i in range(X.shape[0]):
    output.write(str(y[i]))
    output.write(" qid:"+str(query_IDs[i])+" ")
    output.write(" ".join(["{:}:{:}".format(j, Xi) for j, Xi in enumerate(X[i])]))
    output.write(" # info\n")
output.close()
