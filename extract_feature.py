# coding: utf-8
# In[16]:
'''
The idea is to abstract users and items into groups, and then extract features from action history data.
'''
import pandas as pd
import numpy as np

# parameters

inputData = '../JData/'
outputData = 'data/'

n = 2 # how many days before purchase
samplemult = 1 # ratio of nobuy samples over buy samples 
# In[2]:

# for each special action, extract features of n days from all the actions, users and items are abstracted into groups
# This function is for test set
def special_predict(n, df_action_all, df_action_special, dict_user_cat, dict_sku_cat, filename):
    '''
    input：
    n：days before purchase
    df_action_all：action history
    df_action_special: purchase action 
    output:
    df_per: features per special action
    df_action_special_index: original indexes of special actions
    '''

    array_user_id = np.array(df_action_special['user_id'])
    array_sku_id = np.array(df_action_special['sku_id'])

    time = '2016-04-11 00:00:00'
    time_s_string = '2016-04-09 00:00:00'
    df_action_all = df_action_all[(df_action_all['time'] > time_s_string) & (df_action_all['time'] <= time)]

    df_per = pd.DataFrame(columns=('browser', 'addchar', 'delchar', 'buy', 'fav', 'click', 'user_cat', 'sku_cat'))

    # for each special action, build features 
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
        # filter data
        df_action_p = df_action_all[df_action_all['user_sku'] == user_sku]
        # print len(df_action_nobuy)
        # print len(df_action_nobuy)
        df_action_type_counts = df_action_p['type'].value_counts()
        # fillna
        for k in range(1,7):
            try:
                df_action_type_counts[k]
            except:
                df_action_type_counts[k] = 0
        # write a row of features
        df_per.loc[i]={'user_cat':user_cat,'sku_cat':sku_cat,'browser':df_action_type_counts[1],'addchar':df_action_type_counts[2],'delchar':df_action_type_counts[3],'buy':df_action_type_counts[4],'fav':df_action_type_counts[5],'click':df_action_type_counts[6]}
    return df_per

# In[3]:

# for each special action, extract features of n days from all the actions, users and items are abstracted into groups
# This function is for training set
def special_per(n, df_action_all, df_action_special, dict_user_cat, dict_sku_cat, filename):
    '''
    input：
    n：days before purchase
    df_action_all：action history
    df_action_special: purchase action 
    output:
    df_per: features per special action
    df_action_special_index: original indexes of special actions
    '''

    array_user_id = np.array(df_action_special['user_id'])
    array_sku_id = np.array(df_action_special['sku_id'])
    array_time = np.array(df_action_special['time'])

    df_per = pd.DataFrame(columns=('browser', 'addchar', 'delchar', 'buy', 'fav', 'click', 'user_cat', 'sku_cat'))

    # for each special action, build features 
    for i in range(len(df_action_special)):
        print i
        user = array_user_id[i]
        # print user,df1[df1['user_id'] == user]
        user_cat = dict_user_cat[user]
        sku = array_sku_id[i]
        # print sku,df_pb[df_pb['sku_id'] == sku]
        sku_cat = dict_sku_cat[sku]
        time = array_time[i]
        time_s_datetime = pd.datetime.strptime(time , '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = n)
        time_s_string = pd.datetime.strftime(time_s_datetime , '%Y-%m-%d %H:%M:%S')
        # filter data
        df_action_p = df_action_all[(df_action_all['user_id'] == user) & (df_action_all['sku_id'] == sku) & (df_action_all['time'] > time_s_string) & (df_action_all['time'] <= time)]
        # save indexes of special actions
        if i > 0:
            df_action_special_index = df_action_special_index.append(df_action_p.index)
        else:
            df_action_special_index = df_action_p.index
        # print len(df_action_nobuy)
        # print len(df_action_nobuy)
        df_action_type_counts = df_action_p['type'].value_counts()
        # fillna
        for k in range(1,7):
            try:
                df_action_type_counts[k]
            except:
                df_action_type_counts[k] = 0
        # write a row of features
        df_per.loc[i]={'user_cat':user_cat,'sku_cat':sku_cat,'browser':df_action_type_counts[1],'addchar':df_action_type_counts[2],'delchar':df_action_type_counts[3],'buy':df_action_type_counts[4],'fav':df_action_type_counts[5],'click':df_action_type_counts[6]}
    df_per.to_csv(filename)
    df_action_special_index = df_action_special_index.drop_duplicates()
    return df_per, df_action_special_index


# In[4]:

# -----Functions defined-------


# In[5]:


# Files
ACTION_201602_FILE = inputData+"JData_Action_201602.csv"
ACTION_201603_FILE = inputData+"JData_Action_201603.csv"
ACTION_201604_FILE = inputData+"New_JData_Action_201604.csv"
COMMENT_FILE = inputData + "JData_Comment.csv"
PRODUCT_FILE = inputData + "JData_Product.csv"
USER_FILE = inputData + "JData_User.csv"
USER_TABLE_FILE = inputData + "JData_Table_User.csv"
PRODUCT_TABLE_FILE = inputData + "JData_Table_Product.csv"
BEHAVIOR_TABLE_FILE = inputData + "JData_Table_Behavior.csv"



# In[6]:


df_user = pd.read_csv(USER_FILE)
# process user ages
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


# In[7]:

# abstract users into 93 groups
def age_sex_string(inputt):
    try:
        str(int(inputt))
    except:
        return '0'
    else:
        return str(int(inputt))
def regtm_string(inputt):
    inputt = str(inputt)
    try:
        inputt[:4]
    except:
        return '0000'
    else:
        return inputt[:4]
def lvcd_string(inputt):
    return str(inputt)
# df1['user_cat'] = df1['age'].map(age_sex_string) + df1['sex'].map(age_sex_string) + df1['user_lv_cd'].map(lvcd_string) + df1['user_reg_tm'].map(regtm_string)
df1['user_cat'] = df1['age'].map(age_sex_string) + df1['sex'].map(age_sex_string) + df1['user_lv_cd'].map(lvcd_string)


# In[8]:

df_product = pd.read_csv(PRODUCT_FILE)
# print df_product.head(20)

df_comment = pd.read_csv(COMMENT_FILE)
# print df_comment.head()

# merge comments and products
product_behavior = pd.merge(df_product,df_comment, on=['sku_id'], how = 'outer')
# print product_behavior.head(50)
df_pb = product_behavior


# In[9]:

# abstract products into 367 groups
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

# user category dictionary
df_user_cat = df1.loc[:,['user_id','user_cat']]
dict_user_cat = df_user_cat.set_index('user_id')['user_cat'].to_dict()
# product category dictionary
df_sku_cat = df_pb.loc[:,['sku_id','prod_cat']]
df_sku_cat = df_sku_cat.fillna('00000')
dict_sku_cat = df_sku_cat.set_index('sku_id')['prod_cat'].to_dict()

# In[ ]:
    
# save dictionaries
import pickle
def save_obj(obj, name ):
    with open(outputData + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(outputData + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
save_obj(dict_user_cat, 'dict_user_cat')
save_obj(dict_sku_cat, 'dict_sku_cat')

# In[11]:

# action history
df_action_201602 = pd.read_csv(ACTION_201602_FILE)
df_action_201603 = pd.read_csv(ACTION_201603_FILE)
df_action_201604 = pd.read_csv(ACTION_201604_FILE)


# In[12]:

df_action_201602 = df_action_201602.drop_duplicates()
print len(df_action_201602)
# We only need products of category 8 
df_action_201602 = df_action_201602[df_action_201602['cate'] == 8]
print len(df_action_201602)


# In[13]:

df_action_201603 = df_action_201603.drop_duplicates()
print len(df_action_201603)
# We only need products of category 8 
df_action_201603 = df_action_201603[df_action_201603['cate'] == 8]
print len(df_action_201603)


# In[ ]:

df_action_201604 = df_action_201604.drop_duplicates()
print len(df_action_201604)
# We only need products of category 8 
df_action_201604 = df_action_201604[df_action_201604['cate'] == 8]
print len(df_action_201604)


# In[ ]:


df_action_all = df_action_201602
# 'buy' is action type 4
df_action_buy = df_action_all[df_action_all['type'] == 4]
filename_buy = outputData + '201602buy_per.csv'
df_per, df_action_buy_index = special_per(n, df_action_all, df_action_buy, dict_user_cat, dict_sku_cat, filename_buy)
# get features
df_action_all_nobuy = df_action_all.drop(df_action_buy_index)
nobuy_sample_n = len(df_action_buy)
df_action_nobuy = df_action_all.sample(nobuy_sample_n * samplemult)
filename_nobuy = outputData + '201602nobuy_per.csv'
df_per, df_action_nobuy_index = special_per(n, df_action_all_nobuy, df_action_nobuy, dict_user_cat, dict_sku_cat, filename_nobuy)


# In[ ]:


df_action_all = df_action_201603
# 'buy' is action type 4
df_action_buy = df_action_all[df_action_all['type'] == 4]
filename_buy = outputData + '201603buy_per.csv'
df_per, df_action_buy_index = special_per(n, df_action_all, df_action_buy, dict_user_cat, dict_sku_cat, filename_buy)
# get features
df_action_all_nobuy = df_action_all.drop(df_action_buy_index)
nobuy_sample_n = len(df_action_buy)
df_action_nobuy = df_action_all.sample(nobuy_sample_n * samplemult)
filename_nobuy = outputData + '201603nobuy_per.csv'
df_per, df_action_nobuy_index = special_per(n, df_action_all_nobuy, df_action_nobuy, dict_user_cat, dict_sku_cat, filename_nobuy)


# In[ ]:


df_action_all = df_action_201604
# 'buy' is action type 4
df_action_buy = df_action_all[df_action_all['type'] == 4]
filename_buy = outputData + '201604buy_per.csv'
df_per, df_action_buy_index = special_per(n, df_action_all, df_action_buy, dict_user_cat, dict_sku_cat, filename_buy)
# get features
df_action_all_nobuy = df_action_all.drop(df_action_buy_index)
nobuy_sample_n = len(df_action_buy)
df_action_nobuy = df_action_all.sample(nobuy_sample_n * samplemult)
filename_nobuy = outputData + '201604nobuy_per.csv'
df_per_nobuy, df_action_nobuy_index = special_per(n, df_action_all_nobuy, df_action_nobuy, dict_user_cat, dict_sku_cat, filename_nobuy)


# In[ ]:

# save nobuy action of April for prediction
action_nobuy_index = pd.DataFrame(np.array(df_action_nobuy_index))
action_nobuy_index.to_csv(outputData + '201604nobuyindex.csv')
df_action_all_nobuy = df_action_all_nobuy.drop(df_action_nobuy_index)
df_action_all_nobuy.to_csv(outputData + '201604nobuyaction.csv')
print len(df_action_all_nobuy)


# In[ ]:

# -----Features extracted------

# In[ ]:

# concatenate all the features
filename_all = [outputData + '201602nobuy_per.csv', outputData + '201603buy_per.csv', outputData + '201603nobuy_per.csv', outputData + '201604buy_per.csv', outputData + '201604nobuy_per.csv']
df_per_all = pd.read_csv(outputData + '201602buy_per.csv', header = 0, index_col = 0)
for item in filename_all:
    df_per_all = pd.concat([df_per_all, pd.read_csv(item, header = 0, index_col = 0)])
df_per_all.to_csv(outputData + 'df_per_all.csv')




# In[ ]:

df_action_all_nobuy = pd.read_csv(outputData + '201604nobuyaction.csv', header = 0, index_col = 0)


# In[ ]:


df_action_all = df_action_all_nobuy
print len(df_action_all)
# filter out all the users and products for prediction
df_action_all = df_action_all[(df_action_all['user_id'].isin(dict_user_cat.keys())) & (df_action_all['sku_id'].isin(dict_sku_cat.keys()))]
print df_action_all.head()
print len(df_action_all)



# In[ ]:

# combine user_sku as primary key
df_action_all['user_sku'] = df_action_all['user_id']*100000000 + df_action_all['sku_id']
df_user_sku = df_action_all.drop_duplicates(['user_sku'])


# In[ ]:

# prepare to predict for all the user_sku pairs in April 
df_user_sku['time'] = '2016-04-11 00:00:00'
df_user_sku['type'] = 0
print df_user_sku, len(df_user_sku)
df_user_sku.to_csv(outputData + '201604_user_sku.csv')


# In[ ]:

# get features
filename_unknown = outputData + '20160416unknown_per.csv'
df_per_unknown = special_predict(n, df_action_all, df_user_sku, dict_user_cat, dict_sku_cat, filename_unknown)
df_per_unknown.to_csv(filename_unknown)
