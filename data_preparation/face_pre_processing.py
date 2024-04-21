import dlib
import cv2
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import time
import logging
# import matplotlib.pyplot as plt

# img = cv2.imread(image_path) # h,w,c
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = detector(gray)
#
# face = faces[0]
# print(type(face))
# left = face.left()
# top = face.top()
# width = face.width()
# height = face.height()
#
# x, y, w, h = (face.left(), face.top(), face.width(), face.height())
#
# cropped_img = img[y:(y+h),x:(x+w),:]
# plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()


"""
Codes modified from: https://github.com/khanetor/face-alignment-dlib/tree/master
"""

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image(image, det):
    h,w,c = image.shape# modified!!! Handle detected bounding box beyond the original image!
    left, top, right, bottom = rect_to_tuple(det)
    left = max(0,left)
    top = max(0,top)
    right = min(w,right)
    bottom = min(h,bottom)
    return image[top:bottom, left:right]

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset',help='csv file saving dataset information')
    parser.add_argument('--image_path_col', default='image_path')
    parser.add_argument('--output_root', default='/data/qiqitao/FairDeepfakeDetection/filtered_preprocessed_datasets')
    parser.add_argument('--logger',help='logger path')
    args = parser.parse_args()

    LOG_FORMAT = "[%(levelname)s] %(asctime)s: %(message)s"
    logging.basicConfig(filename=args.logger, level=logging.INFO, format=LOG_FORMAT)

    time_anchor = time.time()
    input_dataset = args.input_dataset
    image_path_col = args.image_path_col
    output_root = args.output_root

    # input_dataset = './data_split/realtrain.csv'
    # image_path_col = 'image_path'
    # output_root = '/data/qiqitao/FairDeepfakeDetection/filtered_preprocessed_datasets'

    print('\n')
    print('************************************************')
    print(f'Pre-processing data split: {input_dataset}')

    input_dataset_df = pd.read_csv(input_dataset)
    processed_dataset_df = input_dataset_df.copy()

    raw_images = input_dataset_df.loc[:,image_path_col]

    print(f'Reading input dataset csv file: {time.time()-time_anchor:.2f}s.')
    time_anchor = time.time()
    time_anchor_2 = time.time()

    detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    print(f'Loading face detection and alignment models: {time.time() - time_anchor:.2f}s.')
    time_anchor = time.time()

    logging.info(f'[{time.time()-time_anchor_2:.2f}s] Reading datasets information and loading models.')

    for i,input_image in tqdm(enumerate(raw_images)):
        time_anchor_2=time.time()
        print(f'----------------------\nProcessing image {i+1}/{len(raw_images)}.')
        image_path = input_image.split('FairDeepfakeDetection/')[-1]
        output_image = os.path.join(output_root, image_path)
        if not os.path.exists('/'.join(output_image.split('/')[:-1])):
            os.makedirs('/'.join(output_image.split('/')[:-1]))
        # scale = 1
        print(f'Checking output root exists or not: {time.time() - time_anchor:.2f}s.')
        time_anchor = time.time()

        colored_img = cv2.imread(input_image)
        img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

        print(f'Reading current image: {time.time() - time_anchor:.2f}s.')
        time_anchor = time.time()
        height, width = img.shape[:2]
        # s_height, s_width = height // scale, width // scale
        # img = cv2.resize(img, (s_width, s_height))

        dets = detector(img, 1)
        print(f'Face detection in current image: {time.time() - time_anchor:.2f}s.')
        time_anchor = time.time()

        if not dets: # if no face detected, directly resize the original image
            print(input_image)
            output_img = cv2.resize(img, dsize=(380, 380)) # if no face detected, here the GRAY IMAGE WITH ONLY 1 CHANNEL is saved!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! These gray images should be removed from train/val/test data by check_colored_image.py!!
            print(f'No face is detected, directly resize: {input_image}')
            logging.warning(f'No face detected in {input_image}')
            # cv2.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # raise ValueError('Face detection result is NONE!')
        else:
            det = dets[0].rect  # only extract, crop and align the face with highest confidence in a frame image
            shape = predictor(img, det)
            left_eye = extract_left_eye_center(shape)
            right_eye = extract_right_eye_center(shape)

            M = get_rotation_matrix(left_eye, right_eye)
            rotated = cv2.warpAffine(colored_img, M, (width, height), flags=cv2.INTER_CUBIC)

            print(f'Face alignment in current image: {time.time() - time_anchor:.2f}s.')
            time_anchor = time.time()

            # det can be negative in detection results!!!! Leading to cropping failure!!
            # modify crop_image function.
            cropped = crop_image(rotated, det)
            output_img = cv2.resize(cropped,dsize=(380,380))

            print(f'Face cropping and resizing in current image input dataset csv file: {time.time() - time_anchor:.2f}s.')
            time_anchor = time.time()

        cv2.imwrite(output_image, output_img,[cv2.IMWRITE_PNG_COMPRESSION,0])
        processed_dataset_df.loc[i,'image_path'] = output_image
        if input_dataset.split('/')[:-1]:
            output_dataset = os.path.join('/'.join(input_dataset.split('/')[:-1]),'processed_'+input_dataset.split('/')[-1])
        else:
            output_dataset = 'processed_'+input_dataset

        print(f'Writing current output image: {time.time() - time_anchor:.2f}s.')
        time_anchor = time.time()
        logging.info(f'[{time.time() - time_anchor_2:.2f}s] Pre-processing {i+1}/{len(raw_images)} image: {input_image}.')

    processed_dataset_df.to_csv(output_dataset,index=False)

    print(f'Face extraction, crop, alignment finished: {output_dataset}!')
    logging.info(f'Pre-processing finished: {input_dataset}!')
