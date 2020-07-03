import numpy as np
import matplotlib.pyplot as plt

def accuracy_score(target, predicted, count=False):
    target = np.array(target)
    predicted = np.array(predicted)
    if len(target)!=len(predicted):
        raise ValueError("Both arrays should be of equal length")
    if count:
        return (target==predicted).astype(int).sum()
    return (target==predicted).astype(int).sum()/len(target)

def average_precision(target, predicted):
    target = np.array(target)
    predicted = np.array(predicted)
    rankSorted = np.argsort(-predicted)
    target = target[rankSorted]
    positives = np.where(target==1)[0]
    precisions = (np.arange(positives.shape[0])+1)/(positives+1)
    return precisions.mean()

def confusion_matrix(target, predicted):
    target = np.array(target)
    predicted = np.array(predicted)
    if len(target)!=len(predicted):
        raise ValueError("Both arrays should be of equal length")
    tp = fp = tn = fn = 0
    for i in range(len(target)):
        if target[i] == 1 and predicted[i] == 1:
            tp += 1
        elif target[i] == 1 and predicted[i] == 0:
            fn += 1
        elif target[i] == 0 and predicted[i] == 1:
            fp += 1
        else:
            tn += 1
    return np.array([[tp, fp], [fn, tn]])

def roc(tprs, fprs, label=""):
    f = plt.figure()
    f.suptitle("ROC")
    plt.plot(fprs, tprs, label=label)   
    plt.legend()
    plt.draw()
    plt.pause(0.001)

def auc(target, predicted, report_max_youden=True, plot=True, label=""):
    if len(target) != len(predicted):
        raise ValueError("Unequal target and predictions")
    predictedSorted = np.sort(predicted)
    n = len(predicted)
    i = n-1
    tprs = []
    fprs = []
    thetas = []
    accs = []
    while(i >= 0):
        cfm = confusion_matrix(target, np.array(predicted > predictedSorted[i]).astype(int))
        tp = cfm[0][0]
        fp = cfm[0][1]
        fn = cfm[1][0]
        tn = cfm[1][1]
        accs.append((tp+tn)/(tp+fp+tn+fn))
        tprs.append(tp/(tp+fn))
        fprs.append(fp/(fp+tn))
        thetas.append(predictedSorted[i])
        i-=1
    cfm = confusion_matrix(target, np.array(predicted > predictedSorted[0]-1).astype(int))
    tp = cfm[0][0]
    fp = cfm[0][1]
    fn = cfm[1][0]
    tn = cfm[1][1]
    accs.append((tp+tn)/(tp+fp+tn+fn))
    tprs.append(tp/(tp+fn))
    fprs.append(fp/(fp+tn))
    thetas.append(predictedSorted[0]-1)
    sis = np.argsort(fprs)
    fprs = np.array(fprs)
    fprs.sort()
    tprs = np.array(tprs)[sis]
    thetas = np.array(thetas)[sis]
    accs = np.array(accs)[sis]
    aMax =thetas[np.argmax(accs)]
    auc = np.trapz(tprs, x=fprs)
    if plot:
        roc(tprs, fprs, label=label+" AUC: {0:.2f}".format(auc))
    if report_max_youden:
        return auc, thetas[np.argmax(tprs-fprs)], aMax
    else:
        return auc
if __name__ == "__main__":
    b = [1,0,1,1,1,1,0,0,0,1]
    a = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
    print(average_precision(b, a))
    from sklearn.metrics import average_precision_score
    print(average_precision_score(b, a))