# coding: utf-8

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

inputData = '../JData/'
outputData = 'results/'


n = 2
test_size= .2
samplemult = 1


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
df_p1_buy_per = pd.read_csv('data/df_p1_buy_per.csv',header = 0, index_col = 0)
df_p1_nobuy_per = pd.read_csv('data/df_p1_nobuy_per.csv',header = 0, index_col = 0)
df_p2_buy_per = pd.read_csv('data/df_p2_buy_per.csv',header = 0, index_col = 0)
df_p2_nobuy_per = pd.read_csv('data/df_p2_nobuy_per.csv',header = 0, index_col = 0)

df_per_all = pd.concat([df_p1_buy_per, df_p1_nobuy_per], ignore_index = True)
#df_per_all.sku_cate = df_per_all.sku_cate.astype('str')
df_per_all.drop(['buy8', 'buy4', 'buy2', 'user_cate', 'sku_cate'], axis = 1, inplace = 1)
df_per_all.shape

df_per_test = pd.concat([df_p2_buy_per, df_p2_nobuy_per], ignore_index = True)
df_per_test.drop(['buy8', 'buy4', 'buy2', 'user_cate', 'sku_cate'], axis = 1, inplace = 1)

df_buy_counts = df_per_all['label'].value_counts()
X = np.array(df_per_all.drop('label',axis =1))

y = np.array(df_per_all['label'])

X_unknown = np.array(df_per_test.drop('label',axis =1))

y_true = np.array(df_per_test['label'])
#######################
'''
df_per_all = pd.concat([df_p1_buy_per, df_p1_nobuy_per, df_p2_buy_per, df_p2_nobuy_per], ignore_index = True)
df_per_all.shape
#df_per_all = pd.read_csv(outputData + 'df_per.csv', header = 0, index_col = 0)
df_buy_counts = df_per_all['label'].value_counts()
'''
#for k in range(0, 5):
#    try:
#        df_buy_counts[k]
#    except:
#        df_buy_counts[k] = 0
#for k in range(0, 5):
#    print 'buy ', k, 'count = ', df_buy_counts[k] 
#
## convert the 'buy_number > 1' as 1
#df_per_all[df_per_all.buy > 1] = 1
#
#df_buy_counts = df_per_all['buy'].value_counts()
for k in range(0, 2):
    try:
        df_buy_counts[k]
    except:
        df_buy_counts[k] = 0
for k in range(0, 2):
    print 'buy ', k, 'count = ', df_buy_counts[k] 

X = np.array(df_per_all.drop('label',axis =1))
y = np.array(df_per_all['label'])

## split the train and test data
#sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
#for train_index, test_index in sss.split(X, y):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]

# cv for train data
#skf = StratifiedKFold(y_train, n_folds=5)
skf = StratifiedKFold(y, n_folds=5)

#cls = 'tree' 
#cls = 'bayes' 
#cls = 'GBDT' 
#cls = 'lr' 
#cls = 'svm' 
#cls = 'rf' 
#cls = 'knn' 
model = None
model_name = 'tree'

score = 0
for cls in ['tree', 'bayes', 'GBDT', 'lr', 'rf', 'knn']:
    if cls == 'tree':
      clf = tree.DecisionTreeClassifier()
    if cls == 'bayes':
      clf = naive_bayes.GaussianNB()
    if cls == 'GBDT':
      clf = ensemble.GradientBoostingClassifier()
    if cls == 'lr':
      clf = linear_model.LogisticRegression()
    if cls == 'svm':  
      clf = svm.SVC(kernel = 'linear')
    if cls == 'rf':
      clf = ensemble.RandomForestClassifier(max_depth = 5)
    if cls == 'knn':  
      clf = neighbors.KNeighborsClassifier()
    
    print "cv Performance for " + cls
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    #metric = cv_performance(clf, X_train, y_train, skf)
    metric = cv_performance(clf, X, y, skf)
    print metric_list
    print metric
    if metric[0] > score:
        score = metric[0]
        model = clf
        model_name = cls
print model_name 



clf = model
clf.fit(X, y)


##print list(df_per_all.drop('buy',axis =1).columns)
##print clf.feature_importances_ 
#
## -----testing part------
#
#
#
## -------read features and user_sku pairs to predict -------
#filename_unknown = outputData + '20160416unknown_per.csv'
df_user_sku = pd.read_csv('data/201604_user_sku.csv', header = 0, index_col = 0)
#df_per_unknown = pd.read_csv(filename_unknown, header = 0, index_col = 0)
#
#
## In[ ]:
#
## predict buy action
#X_unknown = np.array(df_per_unknown.drop('buy',axis =1))
#print X_unknown
predictions = clf.predict(X_unknown)
pre_prob = clf.predict_proba(X_unknown)
#print predictions,pre_prob
#
#
print metrics.accuracy_score(y_true, predictions)
print metrics.classification_report(y_true, predictions)
print metrics.confusion_matrix(y_true, predictions)



clf = model
clf.fit(X, y)
# -------read features and user_sku pairs to predict -------
filename_unknown = 'data/20160416unknown_per.csv'
df_user_sku = pd.read_csv('data/201604_user_sku.csv', header = 0, index_col = 0)
df_per_unknown = pd.read_csv(filename_unknown, header = 0, index_col = 0)
df_per_unknown.drop(['user_id', 'sku_id'], axis=1, inplace=True)


# In[ ]:

# predict buy action
X_unknown = np.array(df_per_unknown.drop(['buy8', 'buy4', 'buy2'],axis =1))
#print X_unknown
predictions = clf.predict(X_unknown)
pre_prob = clf.predict_proba(X_unknown)
print predictions,pre_prob


# In[ ]:

# merge predictions into dataframe
df_user_sku['buy'] = predictions
df_user_sku['buy_prob'] = pre_prob[:,1]
#print df_user_sku


# In[ ]:

# filter out purchases
df_buy = df_user_sku[df_user_sku['buy'] == 1]
#print df_buy


# In[ ]:

# get best sku for each user
def best_sku(df):
  return (df.sort(['buy_prob'], ascending = False)).iloc[0,:]


# In[ ]:

# sku group by each user
grouped = df_buy.groupby(['user_id'])
results = grouped.apply(best_sku)


# In[ ]:

def int_to_str(id):
  return str(int(id))

# results
outputData = 'results/'
results = results.loc[:,['user_id','sku_id']]
results['user_id'] = results['user_id'].apply(int_to_str)

resultsfilename = outputData + 'results_n8_5window' + str(n) + '_cls' + cls +'.csv'
results.to_csv(resultsfilename, index=False)
