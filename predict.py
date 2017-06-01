# coding: utf-8

# In[16]:

# 2017年5月16日 整理之前的程序
# 先把用户和产品分别分类后,再和action数据融合
# 数据窗口
n = 9
test_size= .5
cls = 'tree' #0.773114977823
cls = 'bayes' #0.655407710679
cls = 'GBDT' #0.841282838622
cls = 'lr' #0.831252132378
cls = 'svm' #SVM不适用本例
cls = 'rf' #0.797884680996
cls = 'knn' #0.782668031389
resultsfilename = 'results_n' + str(n) + '_cls' + cls +'.csv'
samplemult = 10


# In[2]:

# 函数设定区
# 提取购买行为发生前n天用户行为的特征，用户和产品抽象为类别
#
def special_predict(n, df_action_all, df_action_special, dict_user_cat, dict_sku_cat, filename):
    '''
    输入：
    n：产生特定行为数据之前n天的数据组成特征（特定用户针对特定产品）
    df_action_all：全部行为数据
    df_action_special:特定行为数据
    输出：
    df_per:输出的数据组成特征
    df_action_special_index:构成以上数据组成特征的原始数据索引
    '''
    # df_action_special变为array,加快索引效率
    array_user_id = np.array(df_action_special['user_id'])
    array_sku_id = np.array(df_action_special['sku_id'])
    array_time = np.array(df_action_special['time'])
    time = '2016-04-16 00:00:00'
    # 时间前移10天
    time_s_string = '2016-04-06 00:00:00'
    df_action_all = df_action_all[(df_action_all['time'] > time_s_string) & (df_action_all['time'] <= time)]
    # 建立特征值DataFrame
    df_per = pd.DataFrame(columns=('browser', 'addchar', 'delchar', 'buy', 'fav', 'click', 'user_cat', 'sku_cat'))

    # 循环建立各个specail行为前n天的特征值
    for i in range(len(df_action_special)):
        if i%10000 == 0:
            print i
        user = array_user_id[i]
        # print user,df1[df1['user_id'] == user]
        user_cat = dict_user_cat[user]
        sku = array_sku_id[i]
        # print sku,df_pb[df_pb['sku_id'] == sku]
        sku_cat = dict_sku_cat[sku]
        user_sku = user * 100000000 + sku
        # 筛选购买动作前n天的数据
        df_action_p = df_action_all[df_action_all['user_sku'] == user_sku]
        # print len(df_action_nobuy)
        # 提取特征值,各项动作的次数
        # print len(df_action_nobuy)
        df_action_type_counts = df_action_p['type'].value_counts()
        # 处理异常数据\缺失值
        for k in range(1,7):
            try:
                df_action_type_counts[k]
            except:
                df_action_type_counts[k] = 0
        # 写入一行数据特征值
        df_per.loc[i]={'user_cat':user_cat,'sku_cat':sku_cat,'browser':df_action_type_counts[1],'addchar':df_action_type_counts[2],'delchar':df_action_type_counts[3],'buy':df_action_type_counts[4],'fav':df_action_type_counts[5],'click':df_action_type_counts[6]}
    return df_per


# In[3]:

# 函数设定区
# 提取购买行为发生前n天用户行为的特征，用户和产品抽象为类别
#
def special_per(n, df_action_all, df_action_special, dict_user_cat, dict_sku_cat, filename):
    '''
    输入：
    n：产生特定行为数据之前n天的数据组成特征（特定用户针对特定产品）
    df_action_all：全部行为数据
    df_action_special:特定行为数据
    输出：
    df_per:输出的数据组成特征
    df_action_special_index:构成以上数据组成特征的原始数据索引
    '''
    import pandas as pd
    import numpy as np
    # df_action_special变为array,加快索引效率
    array_user_id = np.array(df_action_special['user_id'])
    array_sku_id = np.array(df_action_special['sku_id'])
    array_time = np.array(df_action_special['time'])

    # 建立特征值DataFrame
    df_per = pd.DataFrame(columns=('browser', 'addchar', 'delchar', 'buy', 'fav', 'click', 'user_cat', 'sku_cat'))

    # 循环建立各个specail行为前n天的特征值
    for i in range(len(df_action_special)):
        print i
        user = array_user_id[i]
        # print user,df1[df1['user_id'] == user]
        user_cat = dict_user_cat[user]
        sku = array_sku_id[i]
        # print sku,df_pb[df_pb['sku_id'] == sku]
        sku_cat = dict_sku_cat[sku]
        time = array_time[i]
        # 时间前移10天
        time_s_datetime = pd.datetime.strptime(time , '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = n)
        time_s_string = pd.datetime.strftime(time_s_datetime , '%Y-%m-%d %H:%M:%S')
        # 筛选购买动作前n天的数据
        df_action_p = df_action_all[(df_action_all['user_id'] == user) & (df_action_all['sku_id'] == sku) & (df_action_all['time'] > time_s_string) & (df_action_all['time'] <= time)]
        # 动作数据集记录以上数据索引,标记为已使用过的数据,以便之后删除
        if i > 0:
            df_action_special_index = df_action_special_index.append(df_action_p.index)
        else:
            df_action_special_index = df_action_p.index
        # print len(df_action_nobuy)
        # 提取特征值,各项动作的次数
        # print len(df_action_nobuy)
        df_action_type_counts = df_action_p['type'].value_counts()
        # 处理异常数据\缺失值
        for k in range(1,7):
            try:
                df_action_type_counts[k]
            except:
                df_action_type_counts[k] = 0
        # 写入一行数据特征值
        df_per.loc[i]={'user_cat':user_cat,'sku_cat':sku_cat,'browser':df_action_type_counts[1],'addchar':df_action_type_counts[2],'delchar':df_action_type_counts[3],'buy':df_action_type_counts[4],'fav':df_action_type_counts[5],'click':df_action_type_counts[6]}
    df_per.to_csv(filename)
    df_action_special_index = df_action_special_index.drop_duplicates()
    return df_per, df_action_special_index


# In[4]:

# -----以上是必备函数区域-------


# In[5]:

# -*- coding = utf-8 -*-
# 设定数据存储文件位置
import pandas as pd
import numpy as np

# -----以下是数据预测部分------

# 读取所有已保存的数据
df_per_all = pd.read_csv('df_per_all.csv', header = 0, index_col = 0)


# In[ ]:

# 转变成sklearn可以使用的数据库
X = np.array(df_per_all.drop('buy',axis =1))
y = np.array(df_per_all['buy'])


# In[ ]:

# 使用以上数据训练DecisionTreeClassifier模型,训练出数据模型
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)


# In[ ]:

if cls == 'tree':
    from sklearn import tree
    clf =tree.DecisionTreeClassifier()
if cls == 'bayes':
    from sklearn import naive_bayes
    clf = naive_bayes.GaussianNB()
if cls == 'GBDT':
    from sklearn import ensemble
    clf = ensemble.GradientBoostingClassifier()


# In[ ]:

clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
pre_prob = clf.predict_proba(X_test)
pre_log_prob = clf.predict_log_proba(X_test)

from sklearn.metrics import  accuracy_score
print accuracy_score(y_test, predictions)


# In[ ]:

# -------以下是预测数据准备------


# In[ ]:

df_action_all_nobuy = pd.read_csv('201604nobuyaction.csv', header = 0, index_col = 0)


# In[ ]:

# 提取2016-04-16到2016-04-20用户行为
# 设置总数据集合
df_action_all = df_action_all_nobuy
print len(df_action_all),len(df_action_201604)
# 筛选出预测用户和预测产品的相关数据
df_action_all = df_action_all[(df_action_all['user_id'].isin(dict_user_cat.keys())) & (df_action_all['sku_id'].isin(dict_sku_cat.keys()))]
print df_action_all.head()
print len(df_action_all)
# 筛选不筛选是一样一样的


# In[ ]:

# 使用用户产品对作为索引
df_action_all['user_sku'] = df_action_all['user_id']*100000000 + df_action_all['sku_id']
df_user_sku = df_action_all.drop_duplicates(['user_sku'])


# In[ ]:

# 补全数据，形成预测对象数据，所有的4月份用户产品对
df_user_sku['time'] = '2016-04-16 00:00:00'
df_user_sku['type'] = 0
print df_user_sku, len(df_user_sku)


# In[ ]:

# 保存用户数据对
df_user_sku.to_csv('201604_user_sku.csv')


# In[ ]:

# 寻找特征数据
filename_unknown = '20160416unknown_per.csv'
df_per_unknown = special_predict(n, df_action_all, df_user_sku, dict_user_cat, dict_sku_cat, filename_unknown)
df_per_unknown.to_csv(filename_unknown)


# In[ ]:

# -------进行数据预测--------
df_user_sku = pd.read_csv('201604_user_sku.csv', header = 0, index_col = 0)
df_per_unknown = pd.read_csv(filename_unknown, header = 0, index_col = 0)


# In[ ]:

# 使用之前的模型进行数据预测
X_unknown = np.array(df_per_unknown.drop('buy',axis =1))
print X_unknown
predictions = clf.predict(X_unknown)
pre_prob = clf.predict_proba(X_unknown)
print predictions,pre_prob


# In[ ]:

# 预测结果和用户产品数据融合，以便接下来处理
df_user_sku['buy'] = predictions
df_user_sku['buy_prob'] = pre_prob[:,1]
print df_user_sku


# In[ ]:

# 筛选出购买结果数据
df_buy = df_user_sku[df_user_sku['buy'] == 1]
print df_buy


# In[ ]:

# 筛选最佳结果 函数
def best_sku(df):
    return (df.sort(['buy_prob'],ascending = False)).iloc[0,:]


# In[ ]:

# 分类统计选取最佳结果
grouped = df_buy.groupby(['user_id'])
results = grouped.apply(best_sku)


# In[ ]:

# 数据转换函数
def int_to_str(id):
    return str(int(id))
	
# 形成数据结果
results = results.loc[:,['user_id','sku_id']]
results['user_id'] = results['user_id'].apply(int_to_str)