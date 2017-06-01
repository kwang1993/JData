# coding: utf-8

# In[16]:

# 2017年5月16日 整理之前的程序
# 先把用户和产品分别分类后,再和action数据融合
# 数据窗口
n = 10 # 天数
test_size=0.3
cls = 'tree' #0.773114977823 
cls = 'bayes' #0.655407710679
cls = 'GBDT' #0.841282838622
cls = 'lr' #0.831252132378
cls = 'svm' #SVM不适用本例
cls = 'rf' #0.797884680996
cls = 'knn' #0.782668031389
resultsfilename = 'results_n' + str(n) + '_cls' + cls +'.csv'
samplemult = 1s


# In[2]:

# 函数设定区
# 提取购买行为发生前n天用户行为的特征，用户和产品抽象为类别
# This is for the test set
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
  # 前10天所有action
  df_action_all = df_action_all[(df_action_all['time'] > time_s_string) & (df_action_all['time'] <= time)]
  # 建立特征值DataFrame
  df_per = pd.DataFrame(columns=('browser', 'addchar', 'delchar', 'buy', 'fav', 'click', 'user_cat', 'sku_cat'))

  # 循环建立各个special行为前n天的特征值
  for i in range(len(df_action_special)):
      if i%10000 == 0:
          print i
      user = array_user_id[i]
      # print user,df1[df1['user_id'] == user]
      user_cat = dict_user_cat[user]
      sku = array_sku_id[i]
      # print sku,df_pb[df_pb['sku_id'] == sku]
      sku_cat = dict_sku_cat[sku]
      # pk = (user_id, sku_id)
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
# This is for the training set
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

# 文件列表
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


# In[6]:

# 用户数据读取及处理
df_user = pd.read_csv(USER_FILE)
# 用户年龄替换成数值
def user_age_map(user_age):
  USER_AGE_MAP = {u'15岁以下'.encode('gbk'): 1,
              u'16-25岁'.encode('gbk'): 2,
              u'26-35岁'.encode('gbk'): 3,
              u'36-45岁'.encode('gbk'): 4,
              u'46-55岁'.encode('gbk'): 5,
              u'56岁以上'.encode('gbk'): 6,
              u'-1'.encode('gbk'): 0}
  try:
      USER_AGE_MAP[user_age]
      # print outputt
  except:
      outputt = user_age
  else:
      outputt = USER_AGE_MAP[user_age]
  # print outputt
  return outputt
# print user_age_map(df_user.iloc[0, 1])
df1 = df_user.applymap(user_age_map)
df1.head(10)
df1.age.value_counts()
df1.sex.value_counts()

# In[7]:

# 聚类用户数据
# 不用,直接根据特征值分类吧,分成93类?
# age or sex to string
def age_sex_string(inputt):
  try:
      str(int(inputt))
  except:
      return '0'
  else:
      return str(int(inputt))
  
# register time to string
def regtm_string(inputt):
  inputt = str(inputt)
  try:
      inputt[:4]
  except:
      return '0000'
  else:
      return inputt[:4]

# user level to string
def lvcd_string(inputt):
  return str(inputt)

# df1['user_cat'] = df1['age'].map(age_sex_string) + df1['sex'].map(age_sex_string) + df1['user_lv_cd'].map(lvcd_string) + df1['user_reg_tm'].map(regtm_string)
df1['user_cat'] = df1['age'].map(age_sex_string) + df1['sex'].map(age_sex_string) + df1['user_lv_cd'].map(lvcd_string)
# Operations between Series (+, -, /, , *) align values based on their associated index values. The result index will be the sorted union of the two indexes.
len(df1.user_cat.value_counts())
# In[8]:

# 产品数据读取及处理
df_product = pd.read_csv(PRODUCT_FILE)
# print df_product.head(20)

# 评价数据
df_comment = pd.read_csv(COMMENT_FILE)
# print df_comment.head()

# 产品和评价数据融合
product_behavior = pd.merge(df_product, df_comment, on=['sku_id'], how = 'outer')
# print product_behavior.head(50)
df_pb = product_behavior


# In[9]:

# 聚类产品数据
# 不用,直接根据特征值分类吧,分成367类
def int_to_string(inputt):
  try:
      str(int(inputt))
  except:
      return '0'
  else:
      return str(int(inputt))
# df1['user_cat'] = df1['age'].map(age_sex_string) + df1['sex'].map(age_sex_string) + df1['user_lv_cd'].map(lvcd_string) + df1['user_reg_tm'].map(regtm_string)
df_pb['prod_cat'] = df_pb['a1'].map(int_to_string) + df_pb['a2'].map(int_to_string) + df_pb['a3'].map(int_to_string) + df_pb['cate'].map(int_to_string) + df_pb['brand'].map(int_to_string)


# In[10]:

# 用户与分类之间建立词典
df_user_cat = df1.loc[:,['user_id','user_cat']]
dict_user_cat = df_user_cat.set_index('user_id')['user_cat'].to_dict()
# 产品与分类之间建立词典
df_sku_cat = df_pb.loc[:,['sku_id','sku_cat']]
df_sku_cat = df_sku_cat.fillna('00000')
dict_sku_cat = df_sku_cat.set_index('sku_id')['sku_cat'].to_dict()



# In[11]:

# 读取行为数据
df_action_201602 = pd.read_csv(ACTION_201602_FILE)
df_action_201603 = pd.read_csv(ACTION_201603_FILE)
df_action_201604 = pd.read_csv(ACTION_201604_FILE)


# In[12]:

# 去除重复行
df_action_201602 = df_action_201602.drop_duplicates()
print len(df_action_201602)
# 去除其他品类数据
df_action_201602 = df_action_201602[df_action_201602['cate'] == 8]
print len(df_action_201602)


# In[13]:

# 去除重复行
df_action_201603 = df_action_201603.drop_duplicates()
print len(df_action_201603)
# 去除其他品类数据
df_action_201603 = df_action_201603[df_action_201603['cate'] == 8]
print len(df_action_201603)


# In[ ]:

# 去除重复行
df_action_201604 = df_action_201604.drop_duplicates()
print len(df_action_201604)
# 去除其他品类数据
df_action_201604 = df_action_201604[df_action_201604['cate'] == 8]
print len(df_action_201604)


# In[ ]:

# 设置总数据集合
df_action_all = df_action_201602
# 提取有购买行为的用户在购买行为发生前n天的特征
df_action_buy = df_action_all[df_action_all['type'] == 4]
filename_buy = '201602buy_per.csv'
df_per, df_action_buy_index = special_per(n, df_action_all, df_action_buy, dict_user_cat, dict_sku_cat, filename_buy)
# 提取无购买行为的用户在购买行为发生前n天的特征
df_action_all_nobuy = df_action_all.drop(df_action_buy_index)
nobuy_sample_n = len(df_action_buy)
df_action_nobuy = df_action_all.sample(nobuy_sample_n * samplemult)
filename_nobuy = '201602nobuy_per.csv'
df_per, df_action_nobuy_index = special_per(n, df_action_all_nobuy, df_action_nobuy, dict_user_cat, dict_sku_cat, filename_nobuy)


# In[ ]:

# 设置总数据集合
df_action_all = df_action_201603
# 提取有购买行为的用户在购买行为发生前n天的特征
df_action_buy = df_action_all[df_action_all['type'] == 4]
filename_buy = '201603buy_per.csv'
df_per, df_action_buy_index = special_per(n, df_action_all, df_action_buy, dict_user_cat, dict_sku_cat, filename_buy)
# 提取无购买行为的用户在购买行为发生前n天的特征
df_action_all_nobuy = df_action_all.drop(df_action_buy_index)
nobuy_sample_n = len(df_action_buy)
df_action_nobuy = df_action_all.sample(nobuy_sample_n * samplemult)
filename_nobuy = '201603nobuy_per.csv'
df_per, df_action_nobuy_index = special_per(n, df_action_all_nobuy, df_action_nobuy, dict_user_cat, dict_sku_cat, filename_nobuy)


# In[ ]:

# 设置总数据集合
df_action_all = df_action_201604
# 提取有购买行为的用户在购买行为发生前n天的特征
df_action_buy = df_action_all[df_action_all['type'] == 4]
filename_buy = '201604buy_per.csv'
df_per, df_action_buy_index = special_per(n, df_action_all, df_action_buy, dict_user_cat, dict_sku_cat, filename_buy)
# 提取无购买行为的用户在购买行为发生前n天的特征
df_action_all_nobuy = df_action_all.drop(df_action_buy_index)
nobuy_sample_n = len(df_action_buy)
df_action_nobuy = df_action_all.sample(nobuy_sample_n * samplemult)
filename_nobuy = '201604nobuy_per.csv'
df_per_nobuy, df_action_nobuy_index = special_per(n, df_action_all_nobuy, df_action_nobuy, dict_user_cat, dict_sku_cat, filename_nobuy)


# In[ ]:

# 保存04月份的未购买数据，这是用于预测的基础数据库
action_nobuy_index = pd.DataFrame(np.array(df_action_nobuy_index))
action_nobuy_index.to_csv('201604nobuyindex.csv')
df_action_all_nobuy = df_action_all_nobuy.drop(df_action_nobuy_index)
df_action_all_nobuy.to_csv('201604nobuyaction.csv')
print len(df_action_all_nobuy)


# In[ ]:

# -----以上是数据处理部分------
# -----以下是数据预测部分------


# In[ ]:

# 训练数据比例


# In[ ]:

# 读取所有已保存的数据
filename_all = ['201602nobuy_per.csv', '201603buy_per.csv', '201603nobuy_per.csv', '201604buy_per.csv', '201604nobuy_per.csv']
df_per_all = pd.read_csv('201602buy_per.csv', header = 0, index_col = 0)
for item in filename_all:
  df_per_all = pd.concat([df_per_all, pd.read_csv(item, header = 0, index_col = 0)])
df_per_all.to_csv('df_per_all.csv')

# In[ ]:

# Since the number of negative samples is 10 times that of positive samples, 
# We do down-sampling or over-sampling

#df_per_all = pd.read_csv('df_per_all.csv', header = 0, index_col = 0)
#df_per_all.head(10)
#df_per_all.loc[df_per_all.buy > 1, ['buy']] = 1
#df_per_all.buy.value_counts()


#
#downSampling = 1 # Choose which sampling method to use
#
#def down_sampling(df):
#    df_buy = df[df['buy'] != 0]
#    df_nobuy = df[df['buy'] == 0].sample(df_buy.shape[0])
#    return pd.concat([df_buy, df_nobuy], ignore_index = True)
#
#def over_sampling(df):
#    df_nobuy = df[df['buy'] == 0] 
#    df_buy = df[df['buy'] != 0].sample(df_nobuy.shape[0], replace=True)
#    return pd.concat([df_buy, df_nobuy], ignore_index = True)   
#
#if downSampling:
#    df_per_all = down_sampling(df_per_all)
#else:
#    df_per_all = over_sampling(df_per_all)
#df_per_all.buy.value_counts()
# In[ ]:

# 转变成sklearn可以使用的数据库
X = np.array(df_per_all.drop('buy',axis =1))
y = np.array(df_per_all['buy'])


# In[ ]:

# 使用以上数据训练DecisionTreeClassifier模型,训练出数据模型
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
X_train.shape
X_test.shape

# In[ ]:
#cls = 'tree' #0.773114977823 
#cls = 'bayes' #0.655407710679
#cls = 'GBDT' #0.841282838622
#cls = 'lr' #0.831252132378
#cls = 'svm' #SVM不适用本例
#cls = 'rf' #0.797884680996
#cls = 'knn' #0.782668031389

from sklearn import metrics

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
# In[ ]:

clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
pre_prob = clf.predict_proba(X_test)
pre_log_prob = clf.predict_log_proba(X_test)

def show_all_metrics_for_multiclass(labels, preds):
    print metrics.accuracy_score(labels, preds)
    print metrics.classification_report(labels, preds)
    print "confusion matrix"
    print metrics.confusion_matrix(labels, preds)
    
show_all_metrics_for_multiclass(y_test, predictions)


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
  return (df.sort(['buy_prob'], ascending = False)).iloc[0,:]


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
results.to_csv(resultsfilename, index=False)