import json
import pandas as pd


gender_weights = {'male': [1, 1],
                # the first element is weight for real samples, the last element is weight for fake samples
                'female': [1, 1]}
race_weights = {'asian': [1, 1],
                'black': [1, 1],
                'white': [1, 1],
                'others': [1, 1]}
intersec_weights = {'female-asian': [1, 1],
                'female-black': [1, 1],
                'female-white': [1, 1],
                'female-others': [1, 1],
                'male-asian': [1, 1],
                'male-black': [1, 1],
                'male-white': [1, 1],
                'male-others': [1, 1]}

gender_weights_modified = {'male': [1, 1],
                # the first element is weight for real samples, the last element is weight for fake samples
                'female': [1, 1]}
race_weights_modified = {'asian': [1, 1],
                'black': [1, 1],
                'white': [1, 1],
                'others': [1, 1]}
intersec_weights_modified = {'female-asian': [1, 1],
                'female-black': [1, 1],
                'female-white': [1, 1],
                'female-others': [1, 1],
                'male-asian': [1, 1],
                'male-black': [1, 1],
                'male-white': [1, 1],
                'male-others': [1, 1]}

with open('/data/qiqitao/FairDeepfakeDetection/data_split/updated_idx_train.json', 'r') as f:
    data = json.load(f)
f.close()

l = len(data)

data_df = pd.DataFrame(data).T
data_df['gender'] = [x.split('-')[0] for x in data_df['intersec_label']]
data_df['race'] = [x.split('-')[1] for x in data_df['intersec_label']]
real_p = data_df['label'].value_counts(0)[0]/l
fake_p = data_df['label'].value_counts(0)[1]/l

# print(real_p)
# print(fake_p)
# assert real_p + fake_p == 1, 'There are labels not being real or fake.'

gender_freq = data_df['gender'].value_counts(normalize=True)
race_freq = data_df['race'].value_counts(normalize=True)
intersec_freq = data_df['intersec_label'].value_counts(normalize=True)

gender_label_counts = pd.crosstab(data_df['label'],data_df['gender']).to_dict()
race_label_counts = pd.crosstab(data_df['label'],data_df['race']).to_dict()
intersec_label_counts = pd.crosstab(data_df['label'],data_df['intersec_label']).to_dict()

gender_weights_subgroup = {'male': [1, 1],
                # the first element is weight for real samples, the last element is weight for fake samples
                'female': [1, 1]}
race_weights_subgroup = {'asian': [1, 1],
                'black': [1, 1],
                'white': [1, 1],
                'others': [1, 1]}
intersec_weights_subgroup = {'female-asian': [1, 1],
                'female-black': [1, 1],
                'female-white': [1, 1],
                'female-others': [1, 1],
                'male-asian': [1, 1],
                'male-black': [1, 1],
                'male-white': [1, 1],
                'male-others': [1, 1]}

for k,v in gender_weights.items():
    v[0] = gender_freq[k] * real_p / (gender_label_counts[k][0] / l)
    v[1] = gender_freq[k] * fake_p / (gender_label_counts[k][1] / l)
for k,v in race_weights.items():
    v[0] = race_freq[k] * real_p / (race_label_counts[k][0] / l)
    v[1] = race_freq[k] * fake_p / (race_label_counts[k][1] / l)
for k, v in intersec_weights.items():
    v[0] = intersec_freq[k] * real_p / (intersec_label_counts[k][0] / l)
    v[1] = intersec_freq[k] * fake_p / (intersec_label_counts[k][1] / l)

for k,v in gender_weights_modified.items():
    v[0] = 0.5 * real_p / (gender_label_counts[k][0] / l)
    v[1] = 0.5 * fake_p / (gender_label_counts[k][1] / l)
for k,v in race_weights_modified.items():
    v[0] = 0.25 * real_p / (race_label_counts[k][0] / l)
    v[1] = 0.25 * fake_p / (race_label_counts[k][1] / l)
for k, v in intersec_weights_modified.items():
    v[0] = 0.125 * real_p / (intersec_label_counts[k][0] / l)
    v[1] = 0.125 * fake_p / (intersec_label_counts[k][1] / l)

for k,v in gender_weights_subgroup.items():
    v[0] = 0.5 / (sum(gender_label_counts[k].values()) / l)
    v[1] = 0.5 / (sum(gender_label_counts[k].values()) / l)
for k,v in race_weights_subgroup.items():
    v[0] = 0.25 / (sum(race_label_counts[k].values()) / l)
    v[1] = 0.25 / (sum(race_label_counts[k].values()) / l)
for k, v in intersec_weights_subgroup.items():
    v[0] = 0.125 / (sum(intersec_label_counts[k].values()) / l)
    v[1] = 0.125 / (sum(intersec_label_counts[k].values()) / l)
print(gender_weights)
print(race_weights)
print(intersec_weights)
#
# print('\n')
print(gender_weights_modified)
print(race_weights_modified)
print(intersec_weights_modified)
#
# print('\n')
# print(gender_weights_subgroup)
# print(race_weights_subgroup)
# print(intersec_weights_subgroup)