import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# def get_files(paths,type_of_file='image'):
#     all_images = []
#     for path in paths:
#         if type_of_file == 'image':
#             img = cv2.imread(path)
#         elif type_of_file == 'mask':
#             img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             ret,img = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
            
#         all_images.append(img)
#     arrays = np.array(all_images)
#     return arrays

def get_files(paths,type_of_file='image'):
    all_images = []
    for path in paths:
        if type_of_file == 'image':
            img = cv2.imread(path)
            img = img.astype('float32') / 255.0
            
        elif type_of_file == 'mask':
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype('float32') / 255.0
            _, img = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # mask = cv2.erode(mask, kernel, iterations=1)
            # mask = cv2.dilate(mask, kernel, iterations=1)
            
        all_images.append(img)
    arrays = np.array(all_images)
    return arrays

def get_folders(list_folders,test_size):

    images_full_list = []
    masks_full_list = []

    for folder in list_folders:
        main_path = str('./datasets/' + folder)
        images_path = str(main_path + '/images/')
        masks_path = str(main_path + '/masks/')
        images_path_sub = os.listdir(images_path)
        masks_path_sub = os.listdir(masks_path)

        images_path_complete = [images_path+x for x in images_path_sub]
        masks_path_complete = [masks_path+x for x in masks_path_sub]

        images_full_list.extend(images_path_complete)
        masks_full_list.extend(masks_path_complete)
    
    images_train, images_test, masks_train, masks_test = train_test_split(
        images_full_list,masks_full_list,test_size=test_size,random_state=42)

    return images_train, images_test, masks_train, masks_test

import numpy as np

def count_pixel_intensity(array):
    # Transforma o array em um array unidimensional
    flat_array = array.ravel()
    
    # Cria um vetor de 256 zeros para armazenar a contagem de pixels de cada intensidade
    count = np.zeros(256, dtype=np.int32)
    
    # Percorre o array unidimensional e conta a quantidade de pixels de cada intensidade
    for i in range(256):
        count[i] = np.sum(flat_array == i)
    
    return count