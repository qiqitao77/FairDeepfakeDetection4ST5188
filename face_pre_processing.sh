cd /data/qiqitai/FairDeepfakeDetection
conda activate fairdeepfake


#python face_pre_processing.py --input_dataset ./data_split/realtrain.csv
python face_pre_processing.py --input_dataset ./data_split/realval.csv
python face_pre_processing.py --input_dataset ./data_split/realtest.csv
python face_pre_processing.py --input_dataset ./data_split/faketrain.csv
python face_pre_processing.py --input_dataset ./data_split/fakeval.csv
python face_pre_processing.py --input_dataset ./data_split/faketest.csv
python face_pre_processing.py --input_dataset filtered_AC-Celeb-DF.csv
python face_pre_processing.py --input_dataset filtered_AC-DFD.csv
