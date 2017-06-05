import numpy as np
import pandas as pd

inputData = '../JData/'
outputData = 'data_1/'

ACTION_201602_FILE = inputData + "JData_Action_201602.csv"
ACTION_201603_FILE = inputData + "JData_Action_201603.csv"
ACTION_201604_FILE = inputData + "JData_Action_201604.csv"
COMMENT_FILE = inputData + "JData_Comment.csv"
PRODUCT_FILE = inputData + "JData_Product.csv"
USER_FILE = inputData + "JData_User.csv"

PERIOD_1 = outputData + 'Period_1.csv'
PERIOD_1_test = outputData + 'Period_1_test.csv'
PERIOD_2 = outputData + 'Period_2.csv'
PERIOD_2_test = outputData + 'Period_2_test.csv'
PERIOD_3 = outputData + 'Period_3.csv'
PERIOD_3_test = outputData + 'Period_3_test.csv'


# load data

df3 = pd.read_csv(ACTION_201603_FILE)
df3 = df3.drop_duplicates()
df3 = df3[df3.cate == 8]
df4 = pd.read_csv(ACTION_201604_FILE)
df4 = df4.drop_duplicates()
df4 = df4[df4.cate == 8]

df = pd.concat([df3, df4], ignore_index = True)

df.to_csv(outputData + 'action34.csv')


# split data

'''
# 从dataSet_path中按照 BEGINDAY, ENDDAY 拆分数据集
    #以2016-03-17到2016-04-05数据   预测2016-04-06到2016-04-10某用户是否下单某商品
    #以2016-03-22到2016-04-10数据   预测2016-04-11到2016-04-15某用户是否下单某商品
    #以2016-03-27到2016-04-15数据   预测2016-04-16到2016-04-20某用户是否下单某商品
    
    # 以2016-03-10到2016-04-05数据   预测2016-04-06到2016-04-10某用户是否下单某商品
    # 以2016-03-15到2016-04-10数据   预测2016-04-11到2016-04-15某用户是否下单某商品
    # 以2016-03-20到2016-04-15数据   预测2016-04-16到2016-04-20某用户是否下单某商品
'''
df.info()

df_p1 = df[(df.time >= '2016-03-17 00:00:00') & (df.time < '2016-04-06 00:00:00')]
df_p1_test = df[(df.time >= '2016-04-06 00:00:00') & (df.time < '2016-04-11 00:00:00')]
df_p2 = df[(df.time >= '2016-03-22 00:00:00') & (df.time < '2016-04-11 00:00:00')]
df_p2_test = df[(df.time >= '2016-04-11 00:00:00') & (df.time < '2016-04-16 00:00:00')]
df_p3 = df[(df.time >= '2016-03-27 00:00:00') & (df.time < '2016-04-16 00:00:00')]
df_p3_test = df[(df.time >= '2016-04-16 00:00:00') & (df.time < '2016-04-21 00:00:00')]

df_p1.to_csv(PERIOD_1)
df_p2.to_csv(PERIOD_2)
df_p3.to_csv(PERIOD_3)
df_p1_test.to_csv(PERIOD_1_test)
df_p2_test.to_csv(PERIOD_2_test)


# extract features

import pickle
def save_obj(obj, name ):
    with open(outputData + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(outputData + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
dict_user_cat = load_obj('dict_user_cat')
dict_sku_cat = load_obj('dict_sku_cat')

n = 8
def special_per(n, df_action_all, df_action_special, dict_user_cat, dict_sku_cat, label):
    '''
    input：
    n：days before purchase
    df_action_all：action history
    df_action_special: purchase action 
    output:
    df_per: features per special action

    '''

    # df_action_special变为array,加快索引效率
    array_user_id = np.array(df_action_special['user_id'])
    array_sku_id = np.array(df_action_special['sku_id'])
    array_time = np.array(df_action_special['time'])

    # 建立特征值DataFrame
    df_per = pd.DataFrame(columns=('browser8', 'addchar8', 'delchar8', 'buy8', 'fav8', 'click8', 
                                   'browser4', 'addchar4', 'delchar4', 'buy4', 'fav4', 'click4', 
                                   'browser2', 'addchar2', 'delchar2', 'buy2', 'fav2', 'click2', 
                                   'user_cate', 'sku_cate', 'label'))

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
        time_s_datetime8 = pd.datetime.strptime(time , '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = 8)
        time_s_datetime4 = pd.datetime.strptime(time , '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = 4)
        time_s_datetime2 = pd.datetime.strptime(time , '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = 2)
        time_s_string8 = pd.datetime.strftime(time_s_datetime8 , '%Y-%m-%d %H:%M:%S')
        time_s_string4 = pd.datetime.strftime(time_s_datetime4 , '%Y-%m-%d %H:%M:%S')
        time_s_string2 = pd.datetime.strftime(time_s_datetime2 , '%Y-%m-%d %H:%M:%S')
        # 筛选购买动作前n天的数据
        df_action_p8 = df_action_all[(df_action_all['user_id'] == user) & (df_action_all['sku_id'] == sku) & (df_action_all['time'] > time_s_string8) & (df_action_all['time'] <= time)]
        # 动作数据集记录以上数据索引,标记为已使用过的数据,以便之后删除
        df_action_p4 = df_action_p8[(df_action_p8['time'] > time_s_string4)]
        df_action_p2 = df_action_p4[(df_action_p4['time'] > time_s_string2)]
#        if i > 0:
#            df_action_special_index = df_action_special_index.append(df_action_p8.index)
#        else:
#            df_action_special_index = df_action_p8.index
        # print len(df_action_nobuy)
        # 提取特征值,各项动作的次数
        # print len(df_action_nobuy)
        df_action_type_counts8 = df_action_p8['type'].value_counts()
        # 处理异常数据\缺失值
        for k in range(1,7):
            try:
                df_action_type_counts8[k]
            except:
                df_action_type_counts8[k] = 0

        df_action_type_counts4 = df_action_p4['type'].value_counts()
        # 处理异常数据\缺失值
        for k in range(1,7):
            try:
                df_action_type_counts4[k]
            except:
                df_action_type_counts4[k] = 0

        df_action_type_counts2 = df_action_p2['type'].value_counts()
        # 处理异常数据\缺失值
        for k in range(1,7):
            try:
                df_action_type_counts2[k]
            except:
                df_action_type_counts2[k] = 0
        # 写入一行数据特征值
        df_per.loc[i]={'user_cate':user_cat,'sku_cate':sku_cat,
                       'browser8':df_action_type_counts8[1],'addchar8':df_action_type_counts8[2],'delchar8':df_action_type_counts8[3],'buy8':df_action_type_counts8[4],'fav8':df_action_type_counts8[5],'click8':df_action_type_counts8[6],
                       'browser4':df_action_type_counts4[1],'addchar4':df_action_type_counts4[2],'delchar4':df_action_type_counts4[3],'buy4':df_action_type_counts4[4],'fav4':df_action_type_counts4[5],'click4':df_action_type_counts4[6],
                       'browser2':df_action_type_counts2[1],'addchar2':df_action_type_counts2[2],'delchar2':df_action_type_counts2[3],'buy2':df_action_type_counts2[4],'fav2':df_action_type_counts2[5],'click2':df_action_type_counts2[6],
                       'label':label}
        #print df_per.loc[i]
    #df_per.to_csv(filename)
    #df_action_special_index = df_action_special_index.drop_duplicates()
    return df_per



df_p1_buy = df_p1_test[df_p1_test.type == 4]
df_p1_nobuy = df_p1_test[df_p1_test.type != 4]
df_p1_nobuy = df_p1_nobuy.drop_duplicates(['user_id', 'sku_id'])

df_p1_buy.shape
df_p1_nobuy.shape

df_p1_nobuy = df_p1_nobuy.sample(df_p1_buy.shape[0])


df_p1_buy_per = special_per(n, df_p1, df_p1_buy, dict_user_cat, dict_sku_cat, 1)
df_p1_buy_per.to_csv(outputData + 'df_p1_buy_per.csv')
df_p1_nobuy_per = special_per(n, df_p1, df_p1_nobuy, dict_user_cat, dict_sku_cat, 0)
df_p1_nobuy_per.to_csv(outputData + 'df_p1_nobuy_per.csv')
#df_per1 = pd.concat([df_p1_buy_per, df_p1_nobuy_per], ignore_index = True)
#df_per1.to_csv(outputData + 'df_per1.csv')




df_p2_buy = df_p2_test[df_p2_test.type == 4]
df_p2_nobuy = df_p2_test[df_p2_test.type != 4]
df_p2_nobuy = df_p2_nobuy.drop_duplicates(['user_id', 'sku_id'])

df_p2_buy.shape
df_p2_nobuy.shape

df_p2_nobuy = df_p2_nobuy.sample(df_p2_buy.shape[0])


df_p2_buy_per = special_per(n, df_p2, df_p2_buy, dict_user_cat, dict_sku_cat, 1)
df_p2_buy_per.to_csv(outputData + 'df_p2_buy_per.csv')
df_p2_nobuy_per = special_per(n, df_p2, df_p2_nobuy, dict_user_cat, dict_sku_cat, 0)
df_p2_nobuy_per.to_csv(outputData + 'df_p2_nobuy_per.csv')
#df_per2 = pd.concat([df_p2_buy_per, df_p2_nobuy_per], ignore_index = True)
#df_per2.to_csv(outputData + 'df_per2.csv')
