from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from adaboost import Adaboost
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from evaluation import accuracy_score
from preprocessing import shiftNScale
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from evaluation import roc, average_precision
from sklearn.decomposition import PCA
from cross_validation import KFold



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
X = MinMaxScaler().fit_transform(X)
y = data[:,-1].astype(int)
query_IDs = qrel[:,0].astype(int)
querys = np.arange(401,451)
fold_precs=[]
kf = KFold(5)
for trainQ, testQ in kf.split(querys, querys):
    # np.random.shuffle(querys)
    trainI = []
    for query in trainQ:
        trainI.extend(np.where(query_IDs==query+401)[0])
    trainI = np.array(trainI)
    # testI = np.where(query_IDs>=440)[0]
    np.random.shuffle(trainI)
    # kf = StratifiedKFold(n_splits=5, shuffle=True)

    # for trainI, testI in kf.split(X, y):
    # clf = DecisionTreeClassifier(max_depth=2)
    # clf = GradientBoostingClassifier()
    clf = ExtraTreesClassifier(n_estimators=50, max_depth=10, class_weight="balanced") #0.15
    # clf = AdaBoostClassifier(n_estimators=10, base_estimator=DecisionTreeClassifier(max_depth=2)) #0.14
    # clf = SVC(class_weight="balanced", tol=0.001, C=2, gamma='scale', probability=True)
    # clf = GaussianNB()
    # ones = len(y[trainI][np.where(y[trainI]==1)])
    # zeroInds = np.where(y[trainI]==0)[0][:ones]
    # oneInds = np.where(y[trainI] == 1)[0]
    # trainI = np.concatenate((zeroInds,oneInds))
    # print(len(trainI))
    zeros=y[trainI][np.where(y[trainI]==0)].shape[0]/y[trainI].shape[0]
    sampleWeight = np.zeros(y[trainI].shape)
    sampleWeight[np.where(y[trainI] == 0)] = 1-zeros
    sampleWeight[np.where(y[trainI] == 1)] = zeros
    # X_tr = X[trainI]
    # pca = PCA(whiten=False)

    # X_tr = pca.fit_transform(X_tr)
    # i=1
    # while pca.explained_variance_ratio_[:i].sum() < 0.999:
    #     i += 1
    # X_tr = np.dot(X_tr, pca.components_.T[:,:i])
    # X_te = np.dot(X[testI], pca.components_.T[:,:i])
    clf.fit(X[trainI], y[trainI])
    precs = []
    for i in testQ:
        testI=np.where(query_IDs==i+401)[0]
        pred = clf.predict_proba(X[testI])
        pred[np.where(pred==0)] = 1e-10
        pred = np.log(pred[:,1],pred[:,0])
        precs.append(average_precision(y[testI], pred))
    fold_precs.append(np.mean(precs))
    # fpr, tpr, thresholds = roc_curve(y[testI], pred)
    # fscores = []
    # for threshold in thresholds:
    #     tn, fp, fn, tp = confusion_matrix(y[testI], pred>threshold).ravel()
    #     fscores.append(2*tp/(2*tp+fp+fn))
    # roc(tpr, fpr, "roc")
    # aucS = auc(fpr, tpr)
    # print(aucS)
    # optimal = np.argmax(fscores)
    # bestTheta = thresholds[optimal]
    # pred = (pred > bestTheta).astype(int)

    # print(f1_score(y[testI], pred))
    # print(precision_score(y[testI], pred))
    # print(recall_score(y[testI], pred))
    # print(accuracy_score(y[testI], pred))
    # print(clf.score(X[trainI], y[trainI]))
    # print(clf.score(X[testI], y[testI]))
    # print(np.unique(clf.predict(X[testI])))
    # print(clf.score(X[testI], y[testI]))
print(np.mean(fold_precs))