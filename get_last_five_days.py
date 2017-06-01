import numpy as np
import pandas as pd

path = 'JData/'
df = pd.read_csv(path + 'JData_Action_201604.csv')

df.time = df.time.astype('datetime64[ns]')
df.time.max()

start_time = '2016-04-11 00:00:00'
last_five_days = df[df.time > start_time]
last_five_days.to_csv('last_five_days.csv')
JData_Action_201604 = df.drop(last_five_days.index)
JData_Action_201604.to_csv('JData_Action_201604_without_last_five_days.csv')