import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize 
path = './'
users = pd.read_csv(path + 'user_table.csv')
users = users.set_index('user_id')
users.head(10)
users.shape
#users[(users['click_num'] < users['buy_num']) & (users['browse_num'] < users['buy_num'])]
# empty

crawler_user = users[(users['buy_num'] == 0) & ((users['browse_num'] > 500) | (users['click_num'] > 500) | (users['addcart_num'] > 100))]
crawler_user.head(10)
crawler_user.shape
crawler_user.to_csv(path + 'crawler_users.csv')

filtered_users = users.drop(crawler_user.index)
filtered_users.to_csv(path + "filtered_users.csv")

filtered_users.shape[0] + crawler_user.shape[0] == users.shape[0]
#
#users1 = users.fillna(0)
#users2 = normalize(users1, axis = 0)
#users2 = pd.DataFrame(users2, columns = users1.columns) 
#users2[['browse_num', 'addcart_num', 'delcart_num', 'buy_num', 'favor_num']].head(10)
#users1.addcart_num.max()
#users1.buy_num[users1.addcart_num == users1.addcart_num.max()]


