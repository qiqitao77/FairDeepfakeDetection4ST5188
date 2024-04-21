"""
Filter out the data with uncertain demographic attributes.
Split the FF++ dataset into training, validation, testing sets.
Celeb-DF and DFD are used for testing only.
Generate data split csv files.
"""
import numpy as np
import pandas as pd

celebdf = pd.read_csv('/data/qiqitao/FairDeepfakeDetection/DemographicAnnotation/AC-Celeb-DF.csv')
ffpp = pd.read_csv('/data/qiqitao/FairDeepfakeDetection/DemographicAnnotation/AC-FF++.csv')
dfd = pd.read_csv('/data/qiqitao/FairDeepfakeDetection/DemographicAnnotation/AC-DFD.csv')

##### filter FF++
ffpp_json = []
for _, row in ffpp.iterrows():
    if row['male'] == 0:
        continue
    if row['asian'] == 0 or row['white'] == 0 or row['black'] == 0:
        continue
    if row['asian']+row['white']+row['black'] < 0: # -1: one of the 3 races; -3: others.
        if row['label'] == 1:
          forgery_type = row['path'].split('/')[-5]
        else:
          forgery_type = 'real'
        gender = 'male' if row['male'] == 1 else 'female'
        if row['asian']+row['white']+row['black'] == -3:
            race = 'others'
        elif row['asian'] == 1:
            race = 'asian'
        elif row['white'] == 1:
            race = 'white'
        elif row['black'] == 1:
            race = 'black'
        current_data = {'image_path':row['path']
                        ,'label':row['label']
                        ,'ismale':row['male']
                        ,'isasian':row['asian']
                        ,'iswhite':row['white']
                        ,'isblack':row['black']
                        ,'intersec_label':gender+'-'+race
                        ,'spe_label':forgery_type}
        ffpp_json.append(current_data)

ffpp_filtered = pd.DataFrame(ffpp_json)
ffpp_filtered.to_csv('/data/qiqitao/FairDeepfakeDetection/filtered_AC-FF++.csv',index=False)

print(ffpp_filtered.groupby('spe_label').size())

##### Split FF++
real_samples = ffpp_filtered[ffpp_filtered['label']==0]
nt_samples = ffpp_filtered[ffpp_filtered['spe_label']=='NeuralTextures']
df_samples = ffpp_filtered[ffpp_filtered['spe_label']=='Deepfakes']
fs_samples = ffpp_filtered[ffpp_filtered['spe_label']=='FaceSwap']
f2f_samples = ffpp_filtered[ffpp_filtered['spe_label']=='Face2Face']
fsh_samples = ffpp_filtered[ffpp_filtered['spe_label']=='FaceShifter']

real_train = pd.DataFrame(columns=['image_path','label','ismale','isasian','iswhite','isblack','intersec_label','spe_label'])
fake_train = pd.DataFrame(columns=['image_path','label','ismale','isasian','iswhite','isblack','intersec_label','spe_label'])
real_val = pd.DataFrame(columns=['image_path','label','ismale','isasian','iswhite','isblack','intersec_label','spe_label'])
fake_val = pd.DataFrame(columns=['image_path','label','ismale','isasian','iswhite','isblack','intersec_label','spe_label'])
real_test = pd.DataFrame(columns=['image_path','label','ismale','isasian','iswhite','isblack','intersec_label','spe_label'])
fake_test = pd.DataFrame(columns=['image_path','label','ismale','isasian','iswhite','isblack','intersec_label','spe_label'])

# real_num = len(real_samples)
# real_samples.sort_values(by='image_path')
# real_samples.reset_index(inplace=True,drop=True)
# for _,row in real_samples.iterrows():
#     if _ <= np.floor(real_num * 0.6):
#         real_train.loc[len(real_train)] = row
#         if _ == np.floor(real_num * 0.6):
#             train_last_id = row['image_path'].split('/')[-2]
#     elif _ <= np.floor(real_num * 0.8):
#         if row['image_path'].split('/')[-2] == train_last_id:
#             real_train.loc[len(real_train)] = row
#         else:
#             real_val.loc[len(real_val)] = row
#         if _ == np.floor(real_num * 0.8):
#             val_last_id = row['image_path'].split('/')[-2]
#     else:
#         if row['image_path'].split('/')[-2] == val_last_id:
#             real_val.loc[len(real_val)] = row
#         else:
#             real_test.loc[len(real_test)] = row

def split_dataframe(df,sort_col,id_col,train_percentage,val_percentage,train_set,val_set,test_set):
    num = len(df)
    df.sort_values(by=sort_col)
    df.reset_index(inplace=True, drop=True)
    for _, row in df.iterrows():
        if _ <= np.floor(num * train_percentage):
            train_set.loc[len(train_set)] = row
            if _ == np.floor(num * train_percentage):
                train_last_id = row[id_col].split('/')[-2]
        elif _ <= np.floor(num * (train_percentage+val_percentage)):
            if row[id_col].split('/')[-2] == train_last_id:
                train_set.loc[len(train_set)] = row
            else:
                val_set.loc[len(val_set)] = row
            if _ == np.floor(num * (train_percentage+val_percentage)):
                val_last_id = row[id_col].split('/')[-2]
        else:
            if row[id_col].split('/')[-2] == val_last_id:
                val_set.loc[len(val_set)] = row
            else:
                test_set.loc[len(test_set)] = row
split_dataframe(real_samples,'image_path','image_path', 0.6, 0.2, real_train, real_val,real_test)
split_dataframe(nt_samples,'image_path','image_path', 0.6,0.2, fake_train, fake_val, fake_test)
split_dataframe(df_samples,'image_path','image_path', 0.6,0.2, fake_train, fake_val, fake_test)
split_dataframe(fs_samples,'image_path','image_path', 0.6,0.2, fake_train, fake_val, fake_test)
split_dataframe(f2f_samples,'image_path','image_path', 0.6,0.2, fake_train, fake_val, fake_test)
split_dataframe(fsh_samples,'image_path','image_path', 0.6,0.2, fake_train, fake_val, fake_test)

assert len(real_test)+len(real_val)+len(real_train) == len(real_samples), 'The total number of real samples in train/val/test splits does not match the number of real samples before splitting!'
assert len(fake_test)+len(fake_val)+len(fake_train) == len(nt_samples)+len(df_samples)+len(fsh_samples)+len(fs_samples)+len(f2f_samples), 'The total number of fake samples in train/val/test splits does not match the number of fake samples before splitting!'

real_train.to_csv('./data_split/realtrain.csv',index=False)
real_val.to_csv('./data_split/realval.csv',index=False)
real_test.to_csv('./data_split/realtest.csv',index=False)
fake_train.to_csv('./data_split/faketrain.csv',index=False)
fake_val.to_csv('./data_split/fakeval.csv',index=False)
fake_test.to_csv('./data_split/faketest.csv',index=False)

##### Filter Celeb-DF
celebdf_json = []
for _, row in celebdf.iterrows():
    if row['male'] == 0: # filter out uncertain gender
        continue
    if row['white'] == 0 or row['black'] == 0: # filter out uncertain race
        continue
    if row ['white'] + row['black'] <= 0: # filter out contradictory race annotation (both positive for white and black)
        gender = 'male' if row['male'] == 1 else 'female'
        if row['white'] == 1:
            race = 'white'
        elif row['black'] == 1:
            race = 'black'
        else:
            race = 'others'
        current_data = {'image_path':row['path']
                        ,'label':row['label']
                        ,'ismale':row['male']
                        ,'iswhite':row['white']
                        ,'isblack':row['black']
                        ,'intersec_label':gender+'-'+race
                        }
        celebdf_json.append(current_data)
celebdf_filtered = pd.DataFrame(celebdf_json)
celebdf_filtered.to_csv('filtered_AC-Celeb-DF.csv',index=False)

##### filter DFD
dfd_json = []
for _, row in dfd.iterrows():
    if row['male'] == 0: # filter out uncertain gender
        continue
    if row['white'] == 0 or row['black'] == 0: # filter out uncertain race
        continue
    if row ['white'] + row['black'] <= 0: # filter out contradictory race annotation (both positive for white and black)
        gender = 'male' if row['male'] == 1 else 'female'
        if row['white'] == 1:
            race = 'white'
        elif row['black'] == 1:
            race = 'black'
        else:
            race = 'others'
        current_data = {'image_path':row['path']
                        ,'label':row['label']
                        ,'ismale':row['male']
                        ,'iswhite':row['white']
                        ,'isblack':row['black']
                        ,'intersec_label':gender+'-'+race
                        }
        dfd_json.append(current_data)
dfd_filtered = pd.DataFrame(dfd_json)
dfd_filtered.to_csv('filtered_AC-DFD.csv',index=False)
