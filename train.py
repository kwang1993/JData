# coding: utf-8

n = 9
test_size= .2
cls = 'GBDT' #0.773114977823
#cls = 'bayes' #0.655407710679
#cls = 'GBDT' #0.841282838622
#cls = 'lr' #0.831252132378
#cls = 'svm' #SVM is not suitable here
#cls = 'rf' #0.797884680996
#cls = 'knn' #0.782668031389
resultsfilename = 'results_n' + str(n) + '_cls' + cls +'.csv'
samplemult = 10

# -*- coding = utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn import tree
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import linear_model 
from sklearn import svm 
from sklearn import neighbors

def performance(y_true, y_pred):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions       
    
    Returns
    --------------------
        metric -- ndarray, performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity' 
    """
    
    m = metrics.confusion_matrix(y_true, y_pred)
    #print m
    metric = np.zeros(6)
    metric[0] = metrics.accuracy_score(y_true, y_pred)
    metric[1] = metrics.f1_score(y_true, y_pred) 
    metric[2] = metrics.roc_auc_score(y_true, y_pred)
    metric[3] = metrics.precision_score(y_true, y_pred)   
    
    TP = m[0, 0]
    FN = m[0, 1]
    metric[4] = float(TP) / float(TP + FN)

    TN = m[1, 1]
    FP = m[0, 1]
    metric[5] = float(TN) / float(FP + TN)

    return metric

def cv_performance(clf, X, y, kf):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
    
    Returns
    --------------------
        metric -- ndarray, performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity' 
    """
    metric = np.zeros(6)
    num = 0.0
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        metric = metric + performance(y_test, clf.predict(X_test))
        num += 1
    metric = metric / num
    return metric

def test_performance(clf, X_train, y_train, X_test, y_test, kf):
    clf.fit(X_train, y_train)
    metric = performance(y_test, clf.predict(X_test))
    return metric

# -----training part------

# read the stored training data
df_per_all = pd.read_csv('data/df_per_all.csv', header = 0, index_col = 0)
df_buy_counts = df_per_all['buy'].value_counts()

for k in range(0, 5):
    try:
        df_buy_counts[k]
    except:
        df_buy_counts[k] = 0
for k in range(0, 5):
    print('buy ', k, 'count = ', df_buy_counts[k])

# convert the 'buy_number > 1' as 1
df_per_all[df_per_all.buy > 1] = 1

df_buy_counts = df_per_all['buy'].value_counts()
for k in range(0, 2):
    try:
        df_buy_counts[k]
    except:
        df_buy_counts[k] = 0
for k in range(0, 2):
    print('buy ', k, 'count = ', df_buy_counts[k])

X = np.array(df_per_all.drop('buy',axis =1))
y = np.array(df_per_all['buy'])

# split the train and test data
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# cv for train data
skf = StratifiedKFold(y_train, n_folds=5)
if cls == 'tree':
  from sklearn import tree
  clf = tree.DecisionTreeClassifier()
if cls == 'bayes':
  from sklearn import naive_bayes
  clf = naive_bayes.GaussianNB()
if cls == 'GBDT':
  from sklearn import ensemble
  clf = ensemble.GradientBoostingClassifier()
if cls == 'lr':
  from sklearn import linear_model 
  clf = linear_model.LogisticRegression()
if cls == 'svm':  
  from sklearn import svm 
  clf = svm.SVC(kernel = 'linear')
if cls == 'rf':
  from sklearn import ensemble
  clf = ensemble.randomForestClassifer(max_depth = 5)
if cls == 'knn':  
  from sklearn import neighbors
  clf = neighbors.KNeighborsClassifier()

print "cv Performance for " + cls
metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
metric = cv_performance(clf, X_train, y_train, skf)
print metric_list
print metric

'''
print "test Performance for " + cls
metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
metric = test_performance(clf, X_train, y_train, X_test, y_test, skf)
print metric_list
print metric
'''
