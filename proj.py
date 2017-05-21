import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix 
import implicit



def dataFrameInfo(df):
    print "=================DataFrame Info=================="
    print df
    print df.dtypes
    #print df.describe()    
     
def loadData(path, fname):
    print "=================Load Data=================="
    print "Loading "+ fname +" ..."
    df = pd.read_csv(path + fname, index_col = False)
    print "Loading "+ fname +" finished."  
    dataFrameInfo(df)
    return df

def to_user_item_matrix(df, user_col, item_col): # return csr_matrix
    print "=================Converting to user-item matrix=================="
    users = df[user_col]
    items = df[item_col]
    ones = np.ones(df.shape[0])
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.coo_matrix.html
    # https://en.wikipedia.org/wiki/Sparse_matrix
    mat = coo_matrix((ones, (users, items))) 
    print mat.shape
    print mat.dtype
    return mat.tocsr()

def model_implicit_data(user_item_mat):
    print "=================Model_implicit=================="
    # initialize a model
    model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01)
    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(user_item_mat.transpose())    
    return model

def getRecommendations(model, user_item_mat, userid): # recommend items for a user
    print "================= recommendations: ================="
    recommendations = model.recommend(userid, user_item_mat)
    for i, p in recommendations:
        print i, p
    return recommendations

def getSimilarItems(model, itemid): # find related items
    print "================= Similar items: ================="
    related = model.similar_items(itemid)
    for i, p in related:
        print i, p
    return related
    
def main():
    dataDir = '../JData/'
    
##    listDir = os.listdir(dataDir)
##    listCsv = [fname for fname in listDir if fname[-4:] == '.csv']
##    print listCsv
##    ['JData_Action_201602.csv', 'JData_Action_201603.csv', 'JData_Action_201604.csv',
##     'JData_Comment.csv', 'JData_Product.csv', 'JData_User.csv']

    # load data frame
    df_a2 = loadData(dataDir, 'JData_Action_201602.csv')

    # convert to sparse matrix
    df_a2_order = df_a2[df_a2['type'] == 4].drop(['type'], axis = 1)
    
    mat_a2 = to_user_item_matrix(df_a2_order, 'user_id', 'sku_id')

    # MF
    model = model_implicit_data(mat_a2)
    
    userid = 203632
    recommendations = getRecommendations(model, mat_a2, userid)
    
    itemid = 20077
    related = getSimilarItems(model, itemid)
    
    
if __name__ == '__main__':
    main()
