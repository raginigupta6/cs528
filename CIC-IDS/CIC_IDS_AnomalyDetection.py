import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score


def transform_labels(labels):
    labels1 = []
    for val in labels:
        if val == 'BENIGN':
            labels1.append(+1)
        else:
            labels1.append(-1)
    return labels1


def show_auc(labels, t1, t2):
    return roc_auc_score(labels, t1, multi_class='ovr'), roc_auc_score(labels, t2, multi_class='ovr')


X = pd.read_csv('Sample_train.csv')

features = list(X.keys())[:-1]
# print (len(features))

X = X.to_numpy()
X = X[:, :-1]
# print (X)
# print (np.shape(X))

clf1 = IsolationForest(random_state=0).fit(X)
clf2 = OneClassSVM(gamma='auto').fit(X)

# Test data
Y = pd.read_csv('Sample_test.csv')

Y = Y.to_numpy()
labels = list(Y[:, -1])
labels = transform_labels(labels)
labels[0] = -1
print (labels)

Y = Y[:, :-1]


Y[Y == 'BENIGN'] = 0
print ('Test shape. ', np.shape(Y))

t1, t2 = [], []
for i in range(np.shape(Y)[0]):
    test_data = [float(val) for val in Y[i, :]]
    # print (test_data)
    print ('Data point ', str(i))

    t1.append(clf1.predict([test_data])[0])
    t2.append(clf2.predict([test_data])[0])

print (t1, t2)
print (show_auc(labels, t1, t2))
