import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

ACTION_201602_FILE = "../../JData/JData_Action_201602.csv"
ACTION_201603_FILE = "../../JData/JData_Action_201603.csv"
ACTION_201604_FILE = "../../JData/JData_Action_201604.csv"

def get_from_action_data(fname):
    # drop_duplicates
    df_action = pd.read_csv(fname)

    df_action = df_action.drop_duplicates()
    df_action = df_action[(df_action['cate'] == 8) & (df_action['type'] == 4)]

    return df_action.loc[:, ['user_id', 'sku_id', 'time']]

def plot_week_buy():
    df_ac = get_from_action_data(fname=ACTION_201602_FILE)
    df_ac = df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
    df_ac = df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))
    print(df_ac.dtypes)

    # convert to datatime
    df_ac['time'] = pd.to_datetime(df_ac['time'])

    # convert time to weekday
    df_ac['time'] = df_ac['time'].apply(lambda x: x.weekday() + 1)

    # #(number of buyers) from monday to sunday
    df_user = df_ac.groupby('time')['user_id'].nunique()
    df_user = df_user.to_frame().reset_index()
    df_user.columns = ['weekday', 'user_num']

    # #(number of items bought) from monday to sunday
    df_item = df_ac.groupby('time')['sku_id'].nunique()
    df_item = df_item.to_frame().reset_index()
    df_item.columns = ['weekday', 'item_num']

    # #(number of buy actions) from monday to sunday
    df_ui = df_ac.groupby('time', as_index=False).size()
    df_ui = df_ui.to_frame().reset_index()
    df_ui.columns = ['weekday', 'user_item_num']

    bar_width = 0.2
    opacity = 0.4

    plt.bar(df_user['weekday'], df_user['user_num'], bar_width, 
            alpha=opacity, color='c', label='user')
    plt.bar(df_item['weekday']+bar_width, df_item['item_num'], 
            bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_ui['weekday']+bar_width*2, df_ui['user_item_num'], 
            bar_width, alpha=opacity, color='m', label='user_item')

    plt.xlabel('weekday')
    plt.ylabel('number')
    plt.title('A Week Purchase Table')
    plt.xticks(df_user['weekday'] + bar_width * 3 / 2., (1,2,3,4,5,6,7))
    plt.tight_layout() 
    plt.legend(prop={'size':9})
    plt.show()

def plot_month_buy(fname, month):
    df_ac = get_from_action_data(fname)

    # convert time to day
    df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)

    df_user = df_ac.groupby('time')['user_id'].nunique()
    df_user = df_user.to_frame().reset_index()
    df_user.columns = ['day', 'user_num']

    df_item = df_ac.groupby('time')['sku_id'].nunique()
    df_item = df_item.to_frame().reset_index()
    df_item.columns = ['day', 'item_num']

    df_ui = df_ac.groupby('time', as_index=False).size()
    df_ui = df_ui.to_frame().reset_index()
    df_ui.columns = ['day', 'user_item_num']

    bar_width = 0.2
    opacity = 0.4
    day_range = range(1,len(df_user['day']) + 1, 1)
    plt.figure(figsize=(14,10))

    plt.bar(df_user['day'], df_user['user_num'], bar_width, 
            alpha=opacity, color='c', label='user')
    plt.bar(df_item['day']+bar_width, df_item['item_num'], 
            bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'], 
            bar_width, alpha=opacity, color='m', label='user_item')

    plt.xlabel('day')
    plt.ylabel('number')
    plt.title(month + 'Purchase Table')
    plt.tight_layout() 
    plt.legend(prop={'size':9})
    plt.show()

plot_week_buy()
plot_month_buy(ACTION_201602_FILE, 'Feburaray')
plot_month_buy(ACTION_201603_FILE, 'March')
plot_month_buy(ACTION_201604_FILE, 'April')

