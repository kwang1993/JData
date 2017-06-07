import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

UI_RECORD = "../data/user_buy_record.csv"
BUY_USER_LIST = "../data/buy_user_list.csv"

'''
df_buy = pd.read_csv(UI_RECORD, header=0)
df_buy = df_buy[["user_id", "sku_id", "buy_duration"]]
df_buy = df_buy.drop_duplicates()
df_buy["delta_time"] = pd.to_timedelta(df_buy['buy_duration'])
high_potential = df_buy[df_buy["delta_time"] > pd.Timedelta('1 days 00:00:00')]
print high_potential.head(5)
'''

ui_record = pd.read_csv(UI_RECORD, header=0)
#user_id = 260731
user_id = 285131
item_id = 154636
cu_record = ui_record[(ui_record['user_id'] == user_id) & (ui_record['sku_id'] == item_id)]
cu_record.tail()
time_range = pd.to_datetime(cu_record['time']).map(lambda x: x.strftime('%m-%d %H:%M'))
x_index = range(len(cu_record['type']))

plt.figure(figsize=(18,5))
plt.scatter(x_index, cu_record['type'],c=cu_record['type'], s=36, lw=0, cmap=plt.cm.coolwarm)
plt.plot(x_index, cu_record['type'], 'y--', markersize=1)
plt.xlim(min(x_index) - 1, max(x_index) + 1)
plt.ylim(0, 7)
plt.xlabel('number')
plt.ylabel('behavior')
# plt.xticks(range(len(cu_record['type'])), time_range, rotation='vertical', fontsize=8)
plt.yticks(range(0,8), ["","browse","add cart","del cart","buy","favor", "click"])
plt.tight_layout()
plt.show()
