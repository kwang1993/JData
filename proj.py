import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
def main():
    dataDir = '../JData/'
    
##    listDir = os.listdir(dataDir)
##    listCsv = [fname for fname in listDir if fname[-4:] == '.csv']
##    print listCsv
##    ['JData_Action_201602.csv', 'JData_Action_201603.csv', 'JData_Action_201604.csv',
##     'JData_Comment.csv', 'JData_Product.csv', 'JData_User.csv']

    # load data frame
    df_a2 = loadData(dataDir, 'JData_Action_201602.csv')
    


    
if __name__ == '__main__':
    main()
