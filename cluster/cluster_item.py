import pandas as pd
import numpy as np
from scipy.sparse import *
from scipy import *
from sklearn.cluster import DBSCAN
from sklearn import preprocessing



USER_TABLE = "JData/user_table.csv"
ITEM_TABLE = "JData/item_table.csv"
DF_PER_ALL = "JData/df_per_all.csv"



def add_feature_buy(row, col, feature):
    if float(feature) is not 0.0:
        rows_buy.append(row)
        cols_buy.append(col)
        data_buy.append(feature)
    else:
        rows_buy.append(row)
        cols_buy.append(col)
        data_buy.append(0.0)


def add_feature_nbuy(row, col, feature):
    rows_nbuy.append(row)
    cols_nbuy.append(col)
    data_nbuy.append(feature)


def time_map(time):
    if time < pd.Timedelta('0 days 00:10:00'):
        return 0
    elif time < pd.Timedelta('0 days 0:30:00'):
        return 1
    elif time < pd.Timedelta('0 days 01:00:00'):
        return 2
    elif time < pd.Timedelta('1 days 00:00:00'):
        return 3
    elif time < pd.Timedelta('2 days 00:00:00'):
        return 4
    elif time < pd.Timedelta('5 days 00:00:00'):
        return 5
    elif time < pd.Timedelta('10 days 00:00:00'):
        return 6
    else:
        return 7



cluster_item = pd.read_csv(ITEM_TABLE)


data_buy = []
rows_buy = []
cols_buy = []

data_nbuy = []
rows_nbuy = []
cols_nbuy = []

row_to_index_buy = dict()
row_to_index_nbuy = dict()

jb = 0
jnb = 0

for i, row in enumerate(cluster_item.itertuples(), 1):
    if i % 10000 == 0:
        print i / 10000
    if pd.isnull(getattr(row, "average_delta_time")):   # not buy
        index = int(getattr(row, "sku_id"))
        row_to_index_nbuy[jnb] = index

        a1 = getattr(row, "a1")
        add_feature_nbuy(jnb, 0, a1)

        a2 = getattr(row, "a2")
        add_feature_nbuy(jnb, 1, a2)

        a3 = getattr(row, "a3")
        add_feature_nbuy(jnb, 2, a3)

        jnb += 1

    else:
        index = int(getattr(row, "sku_id"))
        row_to_index_buy[jb] = index

        a1 = getattr(row, "a1")
        add_feature_buy(jb, 0, a1)

        a2 = getattr(row, "a2")
        add_feature_buy(jb, 1, a2)

        a3 = getattr(row, "a3")
        add_feature_buy(jb, 2, a3)

        a4 = time_map(pd.Timedelta(getattr(row, "average_delta_time")))
        add_feature_buy(jb, 3, a4)

        a5 = getattr(row, "buy_num")
        add_feature_buy(jb, 4, a5)

        a6 = getattr(row, "browse_num")
        add_feature_buy(jb, 5, a6)

        a7 = getattr(row, "addcart_num")
        add_feature_buy(jb, 6, a7)

        a8 = getattr(row, "delcart_num")
        add_feature_buy(jb, 7, a8)

        a9 = getattr(row, "favor_num")
        add_feature_buy(jb, 8, a9)

        a10 = getattr(row, "click_num")
        add_feature_buy(jb, 9, a10)

        jb += 1

# print jb, jnb

data_nbuy = np.array(data_nbuy, copy=False)
rows_nbuy = np.array(rows_nbuy, copy=False)
cols_nbuy = np.array(cols_nbuy, copy=False)

data_buy = np.array(data_buy, copy=False)
rows_buy = np.array(rows_buy, copy=False)
cols_buy = np.array(cols_buy, copy=False)


clusterX_nbuy = coo_matrix((data_nbuy, (rows_nbuy, cols_nbuy))).tocsr()
clusterX_buy = coo_matrix((data_buy, (rows_buy, cols_buy))).tocsr()

clusterX_norm_nb = preprocessing.normalize(clusterX_nbuy, axis=0, copy=False).todense()
clusterX_norm_b = preprocessing.normalize(clusterX_buy, axis=0, copy=False).todense()


db_nb = DBSCAN(min_samples=8, eps=0.005).fit(clusterX_norm_nb)
labels = db_nb.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters in items that have not been bought: %d' % n_clusters_)
user_cluster_list = []
for i in range(len(labels)):
    dict = {'user_id': row_to_index_nbuy[i], 'cluster': labels[i]}
    user_cluster_list.append(dict)

df = pd.DataFrame(user_cluster_list)
df.to_csv('nbuy_item_cluster.csv', sep=',')



db_b = DBSCAN(min_samples=8, eps=0.005).fit(clusterX_norm_b)
labels = db_b.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters in items that have been bought: %d' % n_clusters_)
user_cluster_list = []
for i in range(len(labels)):
    dict = {'user_id': row_to_index_buy[i], 'cluster': labels[i]}
    user_cluster_list.append(dict)

df = pd.DataFrame(user_cluster_list)
df.to_csv('buy_item_cluster.csv', sep=',')
