import numpy as np
from math import log2
import time

from ._splitter import Splitter

class DecisionTree:

    class __BinaryTreeNode:
        def __init__(self, data):
            self.data = data
            self.left = None
            self.right = None
        def depth(self):
            if self.data["leaf"]:
                return self.data["depth"]
            return max(self.left.depth(), self.right.depth())

    def depth(self):
        if self.root == None:
            raise ValueError("No tree fitted yet")
        return self.root.depth()

    def __init__(self, min_improvement=0, regression=False, max_depth=None):
        self.__min_improvement = min_improvement
        self.__regression = regression
        self.root = None
        self.__nFeatures = None
        self.__max_depth = max_depth

    def __H(self, y):
        uniq = np.unique(y, return_counts=True)
        p = dict(zip(uniq[0], uniq[1]))
        ret = 0
        total = len(y)
        p.update((k, v/total) for k, v in p.items())
        for label, pi in p.items():
            ret += pi*log2(1/pi)
        return ret


    def __mse(self, y):
        return np.average((y-y.mean())**2)

    def __weightedSum(self, XI, y, fun):
        """Info gain for categorical outputs, weighted mse for numerical outputs
        fun has to be either self.__H or self.__mse"""
        ret = fun(y)
        l = len(XI)
        values = np.unique(XI, return_counts=True)
        for val, count in zip(values[0], values[1]):
            p = count/l
            f = np.where(XI == val)
            yf = y[f]
            ret -= p*fun(yf)
        return ret

    def __chooseFeature(self, X, y):
        mx = -1
        maxI = -1
        maxTheta = -1
        for i in range(X.shape[1]):
            Xs = Splitter().split(X[:, i])
            for XI, theta in Xs:
                ig = self.__weightedSum(XI, y, self.__mse if self.__regression else self.__H)
                if ig > mx:
                    mx = ig
                    maxI = i
                    maxTheta = theta
        return maxI, maxTheta, mx
    
    def __buildTree(self, node, X, y):
        feature, theta, improvement = self.__chooseFeature(X, y)
        if improvement <= self.__min_improvement:
            return
        if self.__max_depth != None and self.__max_depth <= node.data["depth"]:
            return
        node.data["leaf"] = False
        node.data["feature"] = feature
        node.data["theta"] = theta
        fl = np.where(X[:, feature] < theta)
        Xl = X[fl]
        yl = y[fl]
        fr = np.where(X[:, feature] >= theta)
        Xr = X[fr]
        yr = y[fr]
        if self.__regression:
            labelL = yl.mean()
        else:
            labels, count = np.unique(yl, return_counts=True)
            counts = list(zip(labels, count))
            counts.sort(key=lambda x: x[1])
            labelL = counts[-1][0]
        node.left = self.__BinaryTreeNode({"leaf":True, "label":labelL, "depth":node.data["depth"]+1})
        if self.__regression:
            labelR = yr.mean()
        else:
            labels, count = np.unique(yr, return_counts=True)
            counts = list(zip(labels, count))
            counts.sort(key=lambda x: x[1])
            labelR = counts[-1][0]
        node.right = self.__BinaryTreeNode({"leaf":True, "label":labelR, "depth":node.data["depth"]+1})
        self.__buildTree(node.left, Xl, yl)
        self.__buildTree(node.right, Xr, yr)

    def fit(self, X, y):
        s = time.time()
        X = np.array(X)
        y = np.array(y)
        self.__nFeatures = X.shape[1]
        if self.__regression:
            label = y.mean()
        else:
            self.__labels, count = np.unique(y, return_counts=True)
            counts = list(zip(self.__labels, count))
            counts.sort(key=lambda x: x[1])
            label = counts[-1][0]
        self.root = self.__BinaryTreeNode({"leaf":True, "label":label, "depth":0})
        self.__buildTree(self.root, X, y)
        return time.time() - s

    def predict(self, X):
        if self.root == None:
            raise ValueError("No tree has been fitted yet")
        if X.shape[1] != self.__nFeatures:
            raise ValueError("Incorrect test set")
        y = []
        for Xi in X:
            node = self.root
            while not node.data["leaf"]:
                if Xi[node.data["feature"]] < node.data["theta"]:
                    node = node.left
                else:
                    node = node.right
            y.append(node.data["label"])
        return y
            
# tree = DecisionTree()
# X = np.array([[1,1],[1,0],[0,1],[0,0]])
# y = np.array([1,0,0,1])
# tree.fit(X,y)