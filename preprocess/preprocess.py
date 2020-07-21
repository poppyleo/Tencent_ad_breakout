import json
import sys
sys.path.append('/home/none404/hm/Tencent_ads/Tencente_xuan256')
from config import Config
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


config=Config()
str_flag = config.train_type

print(str_flag)

file_path=config.finetune_data+'{}_data/'.format(str_flag)
save_dict = config.save_dict
#保存路径
save_path = file_path + 'processed/'
if  not os.path.exists(save_path):
    os.mkdir(save_path)


train_df = pd.read_csv(file_path+'train.csv')

test_df = pd.read_csv(file_path+'test.csv')

train_df['sequence_len']=train_df['text'].apply(lambda x:x.split(' ').__len__())
test_df['sequence_len']=test_df['text'].apply(lambda x:x.split(' ').__len__())


train_df['all_label']=train_df.apply(lambda row:str(row['gender'])+str(row['age']),axis=1)

train_df['all_label'] = train_df['all_label'].apply(lambda x:save_dict[x])
test_df['all_label'] = -1

print('训练集')
print(train_df['sequence_len'].describe())
print('测试集')
print(test_df['sequence_len'].describe())

# 切分训练集，分成训练集和验证集
print('Train Set Size:', train_df.shape)
new_dev_df = train_df[: 180000]
frames = [train_df[180000:360000], train_df[360000:]]
new_train_df = pd.concat(frames)  # 训练集

new_train_df = new_train_df.fillna('')



new_train_df.to_csv(save_path+'new_train.csv',index=False)
new_dev_df.to_csv(save_path+'new_dev.csv',index=False)

test_df.to_csv(save_path+'new_test.csv',index=False)
# for多卡预测
for jump in range(5):
    start, end = (jump) * 200000, (jump+1) * 200000
    test_df[start:end].to_csv(save_path+'new_test_{}_{}.csv'.format(start,end),index=False)

"""画图"""
# fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,8))
# sns.distplot(train_df[train_df['sequence_len']<=512]['sequence_len'] , ax=ax1, color='blue')
# sns.distplot(test_df[test_df['sequence_len']<=512]['sequence_len'], ax=ax2, color='green')
# # plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# ax1.set_title('TrainDataset')
# ax2.set_title('TestDataset')
# plt.show()
# plt.savefig('savepic_{}'.format(str_flag),dpi = 800,bbox_inches='tight')

"""
保存的字典  第一位为性别
'ad'
{'11': 0,  
 '12': 1,
 '13': 2,
 '14': 3,
 '15': 4,
 '16': 5,
 '17': 6,
 '18': 7,
 '19': 8,
 '110': 9,
 '21': 10,
 '22': 11,
 '23': 12,
 '24': 13,
 '25': 14,
 '26': 15,
 '27': 16,
 '28': 17,
 '29': 18,
 '210': 19
 }
"""
# ad
# 训练集
# count    900000.000000
# mean         89.988656
# std          68.180660
# min          13.000000
# 25%          49.000000
# 50%          70.000000
# 75%         107.000000
# max       20784.000000
# Name: sequence_len, dtype: float64
# 测试集
# count    1000000.000000
# mean          90.131457
# std          148.564116
# min           14.000000
# 25%           49.000000
# 50%           70.000000
# 75%          107.000000
# max       122783.000000
# Name: sequence_len, dtype: float64
# Train Set Size: (900000, 6)
