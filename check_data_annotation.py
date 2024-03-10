"""

Check whether the extracted frames are consistent with the demographic annotations provided in
 "Analyzing Fairness in Deepfake Detection With Massively Annotated Databases" (https://doi.org/10.1109/TTS.2024.3365421).


Author: Qiqi Tao
Date: 2024-03
"""

import os
import pandas as pd
import tqdm

celebdf_anno = pd.read_csv('/data/qiqitao/FairDeepfakeDetection/DemographicAnnotation/A-Celeb-DF.csv')
ffpp_anno = pd.read_csv('/data/qiqitao/FairDeepfakeDetection/DemographicAnnotation/A-FF++.csv')
dfd_anno = pd.read_csv('/data/qiqitao/FairDeepfakeDetection/DemographicAnnotation/A-DFD.csv')

ffpp_anno_copy = ffpp_anno.copy()
dfd_anno_copy = dfd_anno.copy()
celebdf_anno_copy = celebdf_anno.copy()

data_root = '/data/qiqitao/FairDeepfakeDetection'
tmp = '/data/qiqitao/FairDeepfakeDetection/ff++/manipulated_sequences/NeuralTextures/c23/face_images/000_003/101.png'

for i, path in tqdm.tqdm(enumerate(ffpp_anno['path'])):
    path = path.replace('FaceForensics++', 'ff++')
    path = path.replace('raw', 'c23')
    path = path.replace('frame', '')
    path = os.path.join(data_root, path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ffpp_anno_copy.loc[i,'path'] = path

print(f"All annotations in FF++ are consistent with image files!")

for i, path in tqdm.tqdm(enumerate(dfd_anno['path'])):
    path = path.replace('FaceForensics++', 'ff++')
    path = path.replace('raw', 'c23')
    path = path.replace('frame', '')
    path = os.path.join(data_root, path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    dfd_anno_copy.loc[i, 'path'] = path
print(f"All annotations in DFD are consistent with image files!")

for i, path in tqdm.tqdm(enumerate(celebdf_anno['path'])):
    path = path.replace('Celeb-DF-v2_faces_single', 'CelebDF')
    path = path.replace('frame', '')
    l = path.split('/')
    path = os.path.join(data_root, '/'.join(l[:-2]),'face_images','/'.join(l[-2:]))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    celebdf_anno_copy.loc[i, 'path'] = path
print(f"All annotations in Celeb-DF are consistent with image files!")

print(ffpp_anno_copy)
print(dfd_anno_copy)
print(celebdf_anno_copy)

ffpp_anno_copy.to_csv('/data/qiqitao/FairDeepfakeDetection/DemographicAnnotation/AC-FF++.csv', index=False)
dfd_anno_copy.to_csv('/data/qiqitao/FairDeepfakeDetection/DemographicAnnotation/AC-DFD.csv', index=False)
celebdf_anno_copy.to_csv('/data/qiqitao/FairDeepfakeDetection/DemographicAnnotation/AC-Celeb-DF.csv', index=False)
