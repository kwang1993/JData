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
from sklearn import metrics
import matplotlib.pyplot as plt

test_size= .2
n = 8

# -*- coding = utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

def performance(model, y_true, y_pred, y_prob):
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
    metric[2] = metrics.roc_auc_score(y_true, y_prob)
    metric[3] = metrics.precision_score(y_true, y_pred)   
    
    TP = m[0, 0]
    FN = m[0, 1]
    metric[4] = float(TP) / float(TP + FN)

    TN = m[1, 1]
    FP = m[0, 1]
    metric[5] = float(TN) / float(FP + TN)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model)
    plt.legend(loc="lower right")
    plt.show()

    return metric

def cv_performance(model, clf, X, y, kf):
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
    '''
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        metric = metric + performance(y_test, clf.predict(X_test))
        num += 1
    metric = metric / num
    '''
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)
    metric = performance(model, y, clf.predict(X), y_prob[:, 1])
    return metric

def test_performance(clf, X_train, y_train, X_test, y_test, kf):
    clf.fit(X_train, y_train)
    metric = performance(y_test, clf.predict(X_test))
    return metric

# -----training part------

df_user_cat = pd.read_csv('data/user_cluster.csv')
dict_user_cat = df_user_cat.set_index('user_id')['cluster'].to_dict()

df_pb = pd.read_csv('../JData/JData_Product.csv')
def int_to_string(inputt):
    try:
        str(int(inputt))
    except:
        return '0'
    else:
        return str(int(inputt))
# df1['user_cat'] = df1['age'].map(age_sex_string) + df1['sex'].map(age_sex_string) + df1['user_lv_cd'].map(lvcd_string) + df1['user_reg_tm'].map(regtm_string)
df_pb['prod_cat'] = df_pb['a1'].map(int_to_string) + df_pb['a2'].map(int_to_string) + df_pb['a3'].map(int_to_string) + df_pb['cate'].map(int_to_string) + df_pb['brand'].map(int_to_string)
df_sku_cat = df_pb.loc[:,['sku_id','prod_cat']]
df_sku_cat = df_sku_cat.fillna('00000')
dict_sku_cat = df_sku_cat.set_index('sku_id')['prod_cat'].to_dict()

# read the stored training data
df_per_all = pd.read_csv('data/df_per_all.csv', header = 0, index_col = 0)
#df_per_all['user_cat'] = df_per_all['user_id'].map(dict_user_cat)
#df_per_all['item_cat'] = df_per_all['sku_id'].map(dict_sku_cat)
df_per_all.drop(['user_id', 'sku_id'], axis=1, inplace=True)
print df_per_all.head(2)

df_buy_counts = df_per_all['buy8'].value_counts()

# convert the 'buy_number > 1' as 1
df_per_all[df_per_all.buy8 > 1] = 1

df_buy_counts = df_per_all['buy8'].value_counts()
for k in range(0, 2):
    try:
        df_buy_counts[k]
    except:
        df_buy_counts[k] = 0
for k in range(0, 2):
    print('buy ', k, 'count = ', df_buy_counts[k])

df_X = df_per_all.drop(['buy8', 'buy4', 'buy2'],axis =1)
print df_X.columns
X = np.array(df_X)
y = np.array(df_per_all['buy8'])

# split the train and test data
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# cv for train data
#skf = StratifiedKFold(y_train, n_folds=5)
skf = StratifiedKFold(y_train, n_folds=5)

#cls = 'tree' 
#cls = 'bayes' 
#cls = 'GBDT' 
#cls = 'lr' 
#cls = 'svm' 
#cls = 'rf' 
#cls = 'knn' 
model = None
model_name = 'tree'

clf = ensemble.RandomForestClassifier(max_depth = 5)
clf.fit(X, y)
print clf.feature_importances_
features = ['browser8', 'addchar8','delchar8', 'fav8', 'click8', 'browser4',
       'addchar4', 'delchar4', 'fav4', 'click4', 'browser2', 'addchar2',
       'delchar2', 'fav2', 'click2']
fi = clf.feature_importances_
plt.bar(range(len(features)), fi)

plt.xticks(range(len(features)), features)
plt.xticks(rotation = 45)
plt.show()


score = 0
for cls in ['decision tree', 'naive bayes', 'GBDT', 'logistic regression', 'random forest', 'knn']:
    if cls == 'decision tree':
      clf = tree.DecisionTreeClassifier()
    if cls == 'naive bayes':
      clf = naive_bayes.GaussianNB()
    if cls == 'GBDT':
      clf = ensemble.GradientBoostingClassifier()
    if cls == 'logistic regression':
      clf = linear_model.LogisticRegression()
    if cls == 'svm':  
      clf = svm.SVC(kernel = 'linear')
    if cls == 'random forest':
      clf = ensemble.RandomForestClassifier(max_depth = 5)
    if cls == 'knn':  
      clf = neighbors.KNeighborsClassifier()
    
    print "cv Performance for " + cls
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    #metric = cv_performance(clf, X_train, y_train, skf)
    metric = cv_performance(cls, clf, X_train, y_train, skf)
    print metric_list
    print metric
    if metric[0] > score:
        score = metric[0]
        model = clf
        model_name = cls
print model_name # GBDT

'''
print "test Performance for " + model_name
metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
metric = test_performance(model, X_train, y_train, X_test, y_test, skf)
print metric_list
print metric

'''


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

resultsfilename = outputData + 'results_n' + str(n) + '_cls' + cls +'.csv'
results.to_csv(resultsfilename, index=False)

'''

# In[ ]:
    
# evaluation
ground_truth_file = 'ground_truth.csv'
ground_truth = pd.read_csv(outputData + ground_truth_file, index_col = 0)
ground_truth['user_sku'] = ground_truth['user_id']*100000000 + ground_truth['sku_id']
df_user_sku['ground_truth'] = df_user_sku['user_sku'].isin(ground_truth['user_sku'])
#df_user_sku['user_sku'].shape
#ground_truth['user_sku'].shape
print metric_list
print performance(df_user_sku['ground_truth'], df_user_sku['buy'])
print metrics.classification_report(df_user_sku['ground_truth'], df_user_sku['buy'])
print metrics.confusion_matrix(df_user_sku['ground_truth'], df_user_sku['buy'])

Precision = metrics.precision_score(df_user_sku['ground_truth'], df_user_sku['buy'])   
Recall = metrics.recall_score(df_user_sku['ground_truth'], df_user_sku['buy'])
F11 = 6*Recall*Precision/(5*Recall+Precision)
F12 = 5*Recall*Precision/(2*Recall+3*Precision)
Score = 0.4*F11 + 0.6*F12
print Score 


# what about the accuracy of buy/nobuy prediction
buy_or_nobuy = pd.DataFrame({'user_id': df_user_sku['user_id'].unique()})
buy_or_nobuy['buy'] = buy_or_nobuy['user_id'].isin(df_user_sku.user_id[df_user_sku['buy'] == 1])
buy_or_nobuy['ground_truth'] = buy_or_nobuy['user_id'].isin(ground_truth['user_id'])
#buy_or_nobuy['buy'].value_counts()
#buy_or_nobuy['ground_truth'].value_counts()    
buy_or_nobuy = buy_or_nobuy.set_index('user_id') 
buy_or_nobuy.to_csv(outputData + 'evalutation_n' + str(n) + '_cls' + cls +'.csv')  

print metric_list
print performance(buy_or_nobuy['ground_truth'], buy_or_nobuy['buy'])
print metrics.classification_report(buy_or_nobuy['ground_truth'], buy_or_nobuy['buy'])
print metrics.confusion_matrix(buy_or_nobuy['ground_truth'], buy_or_nobuy['buy'])

'''
