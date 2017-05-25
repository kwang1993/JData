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
    print "\nuser number = %d" % len(df['user_id'].unique()) #73299
    print "\nsku number = %d" % len(df['sku_id'].unique()) #20725
    type_counts = df['type'].value_counts()
    print type_counts
    cate_counts = df['cate'].value_counts()
    print cate_counts
     
def loadData(path, fname):
    print "================= Load Data =================="
    print "Loading "+ fname +" ..."
    df = pd.read_csv(path + fname, index_col = False)
    df.drop_duplicates(inplace = True, keep = 'first')
    df['user_id'] = df['user_id'].astype("int64") 
    df['time'] = df['time'].astype("datetime64[ns]")
    # df["time"].dt.month
    print "Loading "+ fname +" finished."  
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

def dfTypeCate(df, type_no, cate_no):
    df_type = df[df['type'] == type_no]
    df_type_cate = df_type[df_type['cate']==cate_no]
    #dataFrameInfo(df_click_8)
    counts = df_type_cate['time'].value_counts().sort_index()
    time_sorted = counts.index
    plt.xticks(rotation = 45)
    plt.plot(time_sorted, counts)
    print "\nuser number = %d" % len(df_type_cate['user_id'].unique())
    print "\nsku number = %d" % len(df_type_cate['sku_id'].unique()) 
    return df_type_cate


def main():
    dataDir = '../JData/'
    
##    listDir = os.listdir(dataDir)
##    listCsv = [fname for fname in listDir if fname[-4:] == '.csv']
##    print listCsv
##    ['JData_Action_201602.csv', 'JData_Action_201603.csv', 'JData_Action_201604.csv',
##     'JData_Comment.csv', 'JData_Product.csv', 'JData_User.csv']

    # load data frame
    df_a2 = loadData(dataDir, 'JData_Action_201602.csv')
    dataFrameInfo(df_a2)
    
    
    df_a2_click_8 = dfTypeCate(df_a2, 6, 8)
    df_a2_click_8['user_id'].value_counts() # group by user
    df_a2_order_8 = dfTypeCate(df_a2, 4, 8)
    df_a2_order_8['user_id'].value_counts().value_counts() # group by counts
    
    df_a3 = loadData(dataDir, 'JData_Action_201603.csv')
    dataFrameInfo(df_a3)
    df_a3_click_8 = dfTypeCate(df_a3, 6, 8)
    df_a3_order_8 = dfTypeCate(df_a3, 4, 8)
    
    df_a4 = loadData(dataDir, 'JData_Action_201604.csv') 
    dataFrameInfo(df_a4)
    df_a4_click_8 = dfTypeCate(df_a4, 6, 8)
    df_a4_order_8 = dfTypeCate(df_a4, 4, 8)


    
  
    '''
    # convert to sparse matrix
    R_order = to_user_item_matrix(df_order['user_id'], df_order['cate'])
    C_order = R_order
    
    
    # MF
    model = model_implicit_data(C_order)
    
    userid = 203632
    recommendations = getRecommendations(model, C_order, userid)
    
    itemid = 20077
    related = getSimilarItems(model, itemid)
    '''


    
if __name__ == '__main__':
    main()
