import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def buy_duration(df_action_all, df_action_special, filename):
    user_sku_dict = {}
    array_user_id = np.array(df_action_special['user_id'])
    array_sku_id = np.array(df_action_special['sku_id'])
    array_time = np.array(df_action_special['time'])
    df_duration = pd.DataFrame(columns=['buy_duration'])
    for i in range(len(df_action_special)):
        #print i
        user = array_user_id[i]
        sku = array_sku_id[i]
        time = array_time[i]
        user_sku = user * 100000000 + sku
        if (user_sku_dict.has_key(user_sku)):
            continue
        else:
            user_sku_dict[user_sku] = 1
        df_action_p = df_action_all[(df_action_all['user_id'] == user) & (df_action_all['sku_id'] == sku) & (df_action_all['time'] <= time)]
        df_action_time = df_action_p.loc[:, 'time']
        buy_duration = pd.datetime.strptime(time , '%Y-%m-%d %H:%M:%S') - pd.datetime.strptime(df_action_time.min(), '%Y-%m-%d %H:%M:%S')
        #print buy_duration
        df_duration.loc[i] = {'buy_duration': buy_duration}
    df_duration.to_csv(filename)
    return df_duration

def buy_duration_to_csv():
    ACTION_201602_FILE = "../../JData/JData_Action_201602.csv"
    ACTION_201603_FILE = "../../JData/JData_Action_201603.csv"
    ACTION_201604_FILE = "../../JData/JData_Action_201604.csv"

    df_action_201602 = pd.read_csv(ACTION_201602_FILE)
    df_action_201603 = pd.read_csv(ACTION_201603_FILE)
    df_action_201604 = pd.read_csv(ACTION_201604_FILE)

    # drop_duplicates
    df_action_201602 = df_action_201602.drop_duplicates()
    print len(df_action_201602)
    df_action_201602 = df_action_201602[df_action_201602['cate'] == 8]
    print len(df_action_201602)
    df_action_201602 = df_action_201602

    df_action_201603 = df_action_201603.drop_duplicates()
    print len(df_action_201603)
    df_action_201603 = df_action_201603[df_action_201603['cate'] == 8]
    print len(df_action_201603)

    df_action_201604 = df_action_201604.drop_duplicates()
    print len(df_action_201604)
    df_action_201604 = df_action_201604[df_action_201604['cate'] == 8]
    print len(df_action_201604)

    df_action_all = df_action_201602
    df_action_buy = df_action_all[df_action_all['type'] == 4]
    df_action_buy = df_action_buy.loc[:, ['user_id', 'sku_id', 'time']]
    print len(df_action_buy)
    filename_buy = '201602buy_duration.csv'
    df_duration = buy_duration(df_action_all, df_action_buy, filename_buy)

    df_action_all = df_action_201603
    df_action_buy = df_action_all[df_action_all['type'] == 4]
    df_action_buy = df_action_buy.loc[:, ['user_id', 'sku_id', 'time']]
    filename_buy = '201603buy_duration.csv'
    df_duration = df_duration.append(buy_duration(df_action_all, df_action_buy, filename_buy))

    df_action_all = df_action_201604
    df_action_buy = df_action_all[df_action_all['type'] == 4]
    df_action_buy = df_action_buy.loc[:, ['user_id', 'sku_id', 'time']]
    filename_buy = '201604buy_duration.csv'
    df_duration = df_duration.append(buy_duration(df_action_all, df_action_buy, filename_buy))
    return df_duration
    #df_duration = df_duration.sort('time')

# df_duration = buy_duration_to_csv()
df_duration = pd.read_csv('buy_duration_results/201602buy_duration.csv', header = 0, index_col = 0)
df_duration = df_duration.append(pd.read_csv('buy_duration_results/201603buy_duration.csv', header = 0, index_col = 0))
df_duration = df_duration.append(pd.read_csv('buy_duration_results/201604buy_duration.csv', header = 0, index_col = 0))

duration_1day = []
time_left = pd.Timedelta('0 days 00:00:00')
num_1day = 0
for i in range(6):
    time_right = time_left + pd.Timedelta('0 days 04:00:00')
    frequency_i = len(df_duration[(pd.to_timedelta(df_duration.buy_duration) > time_left) & (pd.to_timedelta(df_duration.buy_duration) <= time_right)])
    duration_1day.append(frequency_i)
    print '[%s, %s]: %d' % (time_left, time_right, frequency_i)
    time_left = time_right
print duration_1day

duration = []
time_left = pd.Timedelta('0 days 00:00:00')
for i in range(12):
    time_right = time_left + pd.Timedelta('2 days 00:00:00')
    frequency_i = len(df_duration[(pd.to_timedelta(df_duration.buy_duration) > time_left) & (pd.to_timedelta(df_duration.buy_duration) <= time_right)])
    duration.append(frequency_i)
    print '[%s, %s]: %d' % (time_left, time_right, frequency_i)
    time_left = time_right
frequency_last = len(df_duration[(pd.to_timedelta(df_duration.buy_duration) > time_left)])
duration.append(frequency_last)
print duration

bar_left = range(2, 23, 4)
print bar_left
bar_wide = 4
opacity = 0.5
rects1 = plt.bar(bar_left, duration_1day, bar_wide, align='center', alpha=opacity, color='g')
plt.xlabel('Hour')
plt.ylabel('Frequency')
plt.xticks(range(4, 26, 4))
plt.legend()
plt.tight_layout()
plt.show()

bar_left = range(1, 26, 2)
print bar_left
bar_wide = 2
opacity = 0.5
rects1 = plt.bar(bar_left, duration, bar_wide, align='center', alpha=opacity, color='g')
plt.xlabel('Day')
plt.ylabel('Frequency')
plt.xticks(range(2, 25, 2))
plt.legend()
plt.tight_layout()
plt.show()




