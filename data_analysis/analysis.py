import pandas as pd
import numpy as np
from collections import Counter

ACTION_201602_FILE = "../../JData/JData_Action_201602.csv"
ACTION_201603_FILE = "../../JData/JData_Action_201603.csv"
ACTION_201604_FILE = "../../JData/JData_Action_201604.csv"
COMMENT_FILE = "../../JData/JData_Comment.csv"
PRODUCT_FILE = "../../JData/JData_Product.csv"
USER_FILE = "../../JData/JData_User.csv"
NEW_USER_FILE = "../../JData/JData_User_New.csv"

def explore_comment():
	df_comment = pd.read_csv(COMMENT_FILE, header=0)
	df_comment_item = df_comment.drop_duplicates('sku_id')
	print len(df_comment_item)

	df_cate8_product = pd.read_csv(PRODUCT_FILE, header=0)
	intersect = pd.merge(df_comment_item, df_cate8_product, on='sku_id')
	print len(intersect)

if __name__ == "__main__":
	#explore_comment()
	df_ac2 = pd.read_csv(ACTION_201602_FILE, header=0)
	df_ac2_buy = df_ac2[df_ac2['cate']]
	df_ac2 = df_ac2.groupby(['user_id', 'sku_id'])
	df_ac = df_ac.groupby(["user_id", "sku_id"]).apply(average_duration)
	