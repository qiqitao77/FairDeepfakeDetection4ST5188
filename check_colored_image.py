import pandas as pd
import os
import cv2
import json

data_split_root = './data_split'
faketrain_df = pd.read_csv(os.path.join(data_split_root,'processed_faketrain.csv'))
fakeval_df = pd.read_csv(os.path.join(data_split_root,'processed_fakeval.csv'))
faketest_df = pd.read_csv(os.path.join(data_split_root,'processed_faketest.csv'))
realtrain_df = pd.read_csv(os.path.join(data_split_root,'processed_realtrain.csv'))
realval_df = pd.read_csv(os.path.join(data_split_root,'processed_realval.csv'))
realtest_df = pd.read_csv(os.path.join(data_split_root,'processed_realtest.csv'))

train_df = pd.concat([realtrain_df,faketrain_df]).reset_index(drop=True)
val_df = pd.concat([realval_df,fakeval_df]).reset_index(drop=True)
test_df = pd.concat([realtest_df,faketest_df]).reset_index(drop=True)

df = pd.concat([train_df,val_df,test_df]).reset_index(drop=True)
dict = df.to_dict(orient='index')
print(f'--------------------Number of images: {len(dict)}')
with open('grey_image.json','w') as f:
    grey_image_list = []
    for i in range(len(dict)):
        row = dict[i]
        print(f'{i+1} image.')
        img = cv2.imread(row['image_path'])
        if img.shape[-1] != 3: # find the grey image
            print(f'**********************{row["image_path"]}')
            grey_image_list.append(row['image_path'])
    json.dump(grey_image_list,f)

f.close()
