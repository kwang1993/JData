import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix 
import implicit
from datetime import datetime


def dataFrameInfo(df):
    print "================= DataFrame Info =================="
    print df.head(15)
    print "\nshape:"
    print df.shape
    print "\ndtypes:"
    print df.dtypes
    print "\ninfo:"
    print df.info()
    print "\ndescribe:"
    print df.describe()    
     
def loadData(path, fname):
    print "================= Load Data =================="
    print "Loading "+ fname +" ..."
    df = pd.read_csv(path + fname, index_col = False)
    print "Loading "+ fname +" finished."  
    dataFrameInfo(df)
    return df

def to_user_item_matrix(users, items): # return csr_matrix
    print "================= User-item matrix =================="
    ones = np.ones(len(users)) 
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.coo_matrix.html
    # https://en.wikipedia.org/wiki/Sparse_matrix
    R = coo_matrix((ones, (users, items))) 
    print R.shape
    print R.dtype
    return R.tocsr() # convert coo_matrix to csr_matrix

def model_implicit_data(user_item_mat): # input should be Cui matrix
    print "================= Model implicit data =================="
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
    df_a2 = df_a2.drop_duplicates()
    df_a2['user_id'] = df_a2['user_id'].astype("int64") 
    df_a2['time'] = df_a2['time'].astype("datetime64[ns]")
    # df_a2["time"].dt.month
    dataFrameInfo(df_a2)
    type_counts = df_a2['type'].value_counts()
    print type_counts
    cate_counts = df_a2['cate'].value_counts()
    print cate_counts

    
    df_a2_click = df_a2[df_a2['type'] == 6]
    df_a2_click_8 = df_a2_click[df_a2_click['cate']==8]
    dataFrameInfo(df_a2_click_8)
    clicks = df_a2_click_8['time'].value_counts().sort_index()
    datetime_click = clicks.index
    plt.plot(datetime_click, clicks)
    
    df_a2_order = df_a2[df_a2['type'] == 4]
    df_a2_order_8 = df_a2_order[df_a2_order['cate']==8]
    dataFrameInfo(df_a2_order_8)
    orders = df_a2_order_8['time'].value_counts().sort_index()
    datetime_order = orders.index
    plt.plot(datetime_order, orders)
    
    
    
    '''
    # convert to sparse matrix
    R_a2_order = to_user_item_matrix(df_a2_order['user_id'], df_a2_order['cate'])
    C_a2_order = R_a2_order
    
    
    # MF
    model = model_implicit_data(C_a2_order)
    
    userid = 203632
    recommendations = getRecommendations(model, C_a2_order, userid)
    
    itemid = 20077
    related = getSimilarItems(model, itemid)
    '''


    
if __name__ == '__main__':
    main()
