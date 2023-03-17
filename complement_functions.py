import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def get_files(paths):
    all_images = []
    for path in paths:
        img = cv2.imread(path)
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

