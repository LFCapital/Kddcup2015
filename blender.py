from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

def read_data(file_name):    
    f = open(file_name)
    #ignore header
    f.readline()
    samples = []

    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line]
        samples.append(sample)
    return samples

def load():
    """Conveninence function to load all data as numpy arrays."""
    print "Loading data..."
    filename_train = 'D:\\KDD_RAW_DATA\\train114.csv'
    filename_test = 'D:\\KDD_RAW_DATA\\test114.csv'
    filename_train_y = 'D:\\KDD_RAW_DATA\\truth_train.csv'

    X_train = np.array(read_data(filename_train))
    X_test = np.array(read_data(filename_test))
    y_label = read_data(filename_train_y)
        
    y_train = np.array([x[1] for x in y_label])
    return X_train, y_train, X_test
    
X, y, X_submission = load()

# stacking layer 1
kfold = 15
skf = list(StratifiedKFold(y, kfold))
clfs = [RandomForestClassifier(n_estimators=200, n_jobs=-1, min_samples_leaf = 10, max_features = 0.75, criterion='entropy'),
        LogisticRegression(),
        GradientBoostingClassifier(learning_rate =0.05, max_depth=6, n_estimators=1000,verbose = 1),
        ]

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

'''The key idea is use different folders to build model and predict on the rest folder'''

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    cv_roc = np.zeros((kfold, 1))
    for i, (train, test) in enumerate(skf):
        print "Fold", i
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]   
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:,1]
        cv_roc[i] = roc_auc_score(y_test, y_submission)
        print cv_roc[i]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
    print "Avg roc score is",cv_roc.mean(0)
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
 
# the  second layer use logistic regression
clf = LogisticRegression()
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]

np.savetxt(fname='test_blend.csv', X=dataset_blend_test, fmt='%0.15f')