import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ACTION_201602_FILE = "../../JData/JData_Action_201602.csv"
ACTION_201603_FILE = "../../JData/JData_Action_201603.csv"
ACTION_201604_FILE = "../../JData/New_JData_Action_201604.csv"
COMMENT_FILE = "../../JData/JData_Comment.csv"
PRODUCT_FILE = "../../JData/JData_Product.csv"
USER_FILE = "../../JData/JData_User.csv"
NEW_USER_FILE = "../../JData_User_New.csv"

BUY_USER_LIST_FILE = "../data/buy_user_list.csv"
USER_BUY_RECORD = "../data/user_buy_record.csv"
test_record = "../data/test_record.csv"
UI_RECORD = "../data/user_item_record.csv"
USER_DURATION = "../data/user_duration.csv"
ITEM_DURATION = "../data/item_duration.csv"

# (user - item)pair with purchasing recodes
def buy_user_item(fname):
    df_ac = pd.read_csv(fname, header=0)
    df_ac.drop_duplicates()
    # type = 4, purchase
    df_ac = df_ac[df_ac['type'] == 4][["user_id", "sku_id"]]

    return df_ac

# write (user - item) with purchasing records to csv
def find_buy_user():
    df_ac = []
    df_ac.append(buy_user_item(fname=ACTION_201602_FILE))
    df_ac.append(buy_user_item(fname=ACTION_201603_FILE))
    df_ac.append(buy_user_item(fname=ACTION_201604_FILE))
    
    # merge into a single dataFrame
    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac = df_ac.drop_duplicates()
    # write to csv
    df_ac.to_csv(BUY_USER_LIST_FILE, index=False)


# find all of the action records acoording to a specific (user - item)
def ui_record(fname, ui_pair):
    df_ac = pd.read_csv(fname, header=0)
    df_ac.drop_duplicates()

    df = []
    for index, row in ui_pair.iterrows():
        usr_id = row["user_id"]
        sku_id = row["sku_id"]

        # find all of the action records corresponding to the user-item
        df.append(df_ac[(df_ac["user_id"] == usr_id) &
                        (df_ac["sku_id"] == sku_id)])

    df = pd.concat(df, ignore_index=True)
    return df


# average_duration function:
# calculate the average buy-duration of a (user-item). 
def average_duration(group):
    purchase_time = len(group[group["type"] == 4])
    if (purchase_time == 1):
        last_buy_day = max(group[group["type"] == 4]["date"])
        earliest_behave_day = min(group["date"])
        group["buy_duration"] = last_buy_day - earliest_behave_day
    else:
        purchase_date = group[group["type"] == 4]["date"]
        start_date = min(group["date"])
        #print type(start_date)
        average_duration = pd.Timedelta('0 days 00:00:00')

        idx = 0
        for ind, end_date in purchase_date.iteritems():
            average_duration += (end_date - start_date)
            idx += 1
            if (idx == purchase_time):
                break
            start_date = min(group[group["date"] > end_date]["date"])
        average_duration /= purchase_time
        group["buy_duration"] = average_duration
    return group


def average_delta_time(group):
    all_delta_time = group["delta_time"]
    #print type(all_delta_time)
    group["average_delta_time"] = all_delta_time.mean()
    #print all_delta_time.mean()
    return group

# calculate the average buy duration of each user
# buy duration: The time from the first click to the order
def user_buy_duration():
    # user-item with purchasing record

    ui_pair = pd.read_csv(BUY_USER_LIST_FILE, header=0)

    df_ac = []
    df_ac.append(ui_record(ACTION_201602_FILE, ui_pair))
    print 'ac2 ui_record read finished'
    df_ac.append(ui_record(ACTION_201603_FILE, ui_pair))
    print 'ac3 ui_record read finished'
    df_ac.append(ui_record(ACTION_201604_FILE, ui_pair))
    print 'ac4 ui_record read finished'

    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac = df_ac.drop_duplicates()
    # df_ac.to_csv(UI_RECORD, index=False)
    '''

    '''
    # df_ac = pd.read_csv(UI_RECORD, header=0)
    
    print 'start calculateing the average duration'
    # add date attr
    df_ac['date'] = pd.to_datetime(df_ac['time'])
    df_ac = df_ac.groupby(["user_id", "sku_id"]).apply(average_duration)
    
    # write to csv
    df_ac.to_csv(USER_BUY_RECORD, index=False)

    df_buy = pd.read_csv(USER_BUY_RECORD, header=0)
    df_buy = df_buy[["user_id", "sku_id", "buy_duration"]]
    print len(df_buy)
    df_buy = df_buy.drop_duplicates()
    print len(df_buy)
    df_buy["delta_time"] = pd.to_timedelta(df_buy['buy_duration'])
    print df_buy.head(5)

    df_user = df_buy[["user_id", "delta_time"]]
    df_user = df_user.groupby(['user_id'], as_index=False).apply(average_delta_time)
    df_user = df_user[["user_id", "average_delta_time"]]
    df_user = df_user.drop_duplicates()
    df_user.to_csv(USER_DURATION, index=False)

    df_item = df_buy[["sku_id", "delta_time"]]
    df_item = df_item.groupby(['sku_id'], as_index=False).apply(average_delta_time)
    df_item = df_item[["sku_id", "average_delta_time"]]
    df_item = df_item.drop_duplicates()
    df_item.to_csv(ITEM_DURATION, index=False)

def plot_duration(fname):
    df_duration = pd.read_csv(fname, header=0)
    df_duration['time_delta'] = pd.to_timedelta(df_duration['average_delta_time'])
    duration_1day = []
    time_left = pd.Timedelta('0 days 00:00:00')
    num_1day = 0
    for i in range(24):
        time_right = time_left + pd.Timedelta('0 days 01:00:00')
        frequency_i = len(df_duration[(df_duration['time_delta'] > time_left) & (df_duration['time_delta'] <= time_right)])
        duration_1day.append(frequency_i)
        print '[%s, %s]: %d' % (time_left, time_right, frequency_i)
        time_left = time_right
    print duration_1day

    duration = []
    time_left = pd.Timedelta('0 days 00:00:00')
    for i in range(12):
        time_right = time_left + pd.Timedelta('1 days 00:00:00')
        frequency_i = len(df_duration[(df_duration['time_delta'] > time_left) & (df_duration['time_delta'] <= time_right)])
        duration.append(frequency_i)
        print '[%s, %s]: %d' % (time_left, time_right, frequency_i)
        time_left = time_right
    frequency_last = len(df_duration[(df_duration['time_delta'] > time_left)])
    duration.append(frequency_last)
    print duration

    bar_left = np.arange(0.5, 24, 1)
    print bar_left
    bar_wide = 1
    opacity = 0.5
    rects1 = plt.bar(bar_left, duration_1day, bar_wide, align='center', alpha=opacity, color='g')
    plt.xlabel('Hour')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(1, 25, 1))
    plt.legend()
    plt.tight_layout()
    plt.show()

    bar_left = np.arange(0.5, 13, 1)
    print bar_left
    bar_wide = 1
    opacity = 0.5
    rects1 = plt.bar(bar_left, duration, bar_wide, align='center', alpha=opacity, color='g')
    plt.xlabel('Day')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(1, 14, 1))
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    #find_buy_user()
    #user_buy_duration()
    #plot_duration(USER_DURATION)
    plot_duration(ITEM_DURATION)
