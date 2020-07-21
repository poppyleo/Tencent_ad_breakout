import pandas as pd
from config import Config
import os
config = Config()
result_path = '/home/none404/hm/Tencent_ads/finetune_data/result/model_0.5289_0.9533-18750/'

file_list =os.listdir(result_path)
flag = 0
for file in file_list:
    if 'csv' in file:
        if flag==0:
            flag=1
            df = pd.read_csv(result_path+file)
        else:
            df=df.append(pd.read_csv(result_path+file))
print(df.head())
df[['user_id','predicted_age','predicted_gender']].to_csv(result_path+'{}_submit.csv'.format(result_path.split('/')[-2]),index=False)