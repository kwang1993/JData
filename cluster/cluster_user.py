import pandas as pd
import numpy as np
from scipy.sparse import *
from scipy import *
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import preprocessing



ACTION_201602_FILE = "JData/JData_Action_201602.csv"
ACTION_201603_FILE = "JData/JData_Action_201603.csv"
ACTION_201604_FILE = "JData/JData_Action_201604.csv"
COMMENT_FILE = "JData/JData_Comment.csv"
PRODUCT_FILE = "JData/JData_Product.csv"
USER_FILE = "JData/JData_User.csv"
USER_TABLE_FILE = "JData/JData_Table_User.csv"
PRODUCT_TABLE_FILE = "JData/JData_Table_Product.csv"
BEHAVIOR_TABLE_FILE = "JData/JData_Table_Behavior.csv"
DEMO = "demo.csv"

USER_TABLE = "JData/user_table.csv"
ITEM_TABLE = "JData/item_table.csv"
DF_PER_ALL = "JData/df_per_all.csv"



def add_feature(row, col, feature):
    if float(feature) is not 0.0:
        rows.append(row)
        cols.append(col)
        data.append(feature)


user_list = pd.read_csv(DF_PER_ALL).fillna(0)['user_id'].tolist()
user_set = set([int(i) for i in user_list])

cluster_user = pd.read_csv(USER_TABLE).fillna(0)
# cluster_user = cluster_user.set_index('user_id')


data = []
rows = []
cols = []

row_to_index = dict()

j = 0
for i, row in enumerate(cluster_user.itertuples(), 1):
    if i % 10000 == 0:
        print i / 10000

    index = int(getattr(row, "user_id"))
    if index in user_set:
        row_to_index[j] = index

        # age_col = getattr(row, "age") + 1   # -1 -> 0, 0 -> 1, ... age + 1 -> col
        # rows.append(i)
        # cols.append(age_col)
        # data.append(1)

        # sex_col = getattr(row, "sex")
        # rows.append(i)
        # cols.append(sex_col)
        # data.append(1)
        browse_num = getattr(row, "browse_num")
        add_feature(j, 0, browse_num)

        addcart_num = getattr(row, "addcart_num")
        add_feature(j, 1, addcart_num)

        delcart_num = getattr(row, "delcart_num")
        add_feature(j, 2, delcart_num)

        buy_num = getattr(row, "buy_num")
        add_feature(j, 3, buy_num)

        fav_num = getattr(row, "favor_num")
        add_feature(j, 4, fav_num)

        click_num = getattr(row, "click_num")
        add_feature(j, 5, click_num)

        j += 1

data = np.array(data, copy=False)
rows = np.array(rows, copy=False)
cols = np.array(cols, copy=False)



clusterX = coo_matrix((data, (rows, cols))).tocsr()

# # clusterX = np.log1p(clusterX)
clusterX_norm = preprocessing.normalize(clusterX, axis=0, copy=False).todense()

#
#
# # kmeans = KMeans(n_clusters=2, random_state=2)
# # kmeans.fit(clusterX)
# # plt.subplot(111)
# # ypred = kmeans.predict(clusterX)
#
# # pca = PCA(n_components='mle', svd_solver='auto')
# # pca.fit(clusterX_norm.todense())
# # print pca.n_components_
# # print pca.explained_variance_ratio_
#
db = DBSCAN(min_samples=5, eps=0.002).fit(clusterX_norm)
labels = db.labels_
from collections import Counter
c = Counter(labels)
print(c.items())

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x = np.array(clusterX_norm[:,0].T)
y = np.array(clusterX_norm[:,1].T)
z = np.array(clusterX_norm[:,3].T)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=labels)
plt.show()

# user_label = pd.DataFrame(columns=('user_id', 'cluster'))
#
#
#
# user_cluster_list = []
# for i in range(len(labels)):
#     dict = {'user_id': row_to_index[i], 'cluster': labels[i]}
#     user_cluster_list.append(dict)
#
# df = pd.DataFrame(user_cluster_list)
#
# df.to_csv('user_cluster.csv', sep=',')
