# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:45:56 2021

@author: Jerry
"""

from PIL import Image
from tqdm import tqdm
from skimage import io, color, img_as_ubyte
import os 
import cv2
import nrrd
import glob
import numpy as np
import pandas as pd

# In[] def

def rot_and_flip_for_array(array):
    array = np.rot90(array, 3)
    array = np.flip(array, 1)
    
    return array

def get_pixels_hu_image(image, slope = 1, intercept = -1024):

    # 將超過機器掃描範圍的部分設為 0
    # 通常intercept是 -1024, 經過計算之後空氣大約是 0
    image[image < 0] = 0
    
    # 轉換為Hounsfield units (HU)
        
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int8(intercept)
    
    return np.array(image, dtype=np.int16)


# def convert_soft_tissue_window(images):
    
#     images[images < -125] = 0
#     images[images >= 225] = 255
    
#     # images = ((images - np.min(images)) / (np.max(images) - np.min(images))) * 255
    
#     images = np.array(images, dtype=np.int16)
    
#     return images

def convert_soft_tissue_window(images, level = -6.62177, window = 415.822):
    
    img_min = level - window // 2
    img_max = level + window // 2
    window_image = images.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    window_image = ((window_image - np.min(window_image)) / (np.max(window_image) - np.min(window_image))) * 255
    
    return window_image

def get_patient_id(number, patient_data_dir):
    
    
    
    return patient_id


# In[] Get image folder list

## get classfication information
# csv_path = r'D:\NCKU\Thymoma_classfication\data\Mediastinal tumor_CECT.csv'
# class_info = pd.read_csv(csv_path, encoding = 'big5')[['order', 'Patho_class']].set_index('order')
csv_path = r'D:\NCKU\Thymoma_classfication\Mediastinal-tumor-patient-20211124.xlsx'
class_info = pd.read_excel(csv_path)[['Patient_number', 'Patho_class']].set_index('Patient_number')
class_info.fillna('nan', inplace = True)

###########################################################
for index, value in enumerate(class_info['Patho_class']):
    if value != 'Thymoma':
        class_info.loc[index + 1 , 'Patho_class'] = 'other'
###########################################################

print(class_info.iloc[16][0] == 'nan')

classes = list(np.unique(class_info['Patho_class']))
# classes.remove('nan')


## get patient_list
patient_data_dir = r'D:\NCKU\Thymoma_classfication\data\20210901\Mediastinal tumor'
patient_list = os.listdir(patient_data_dir)

## create save folder
seg_save_path = r'D:\NCKU\Thymoma_classfication\data\seg_label2'
if not os.path.exists(seg_save_path):
    os.makedirs(seg_save_path)

for i_class in classes:
    class_path = os.path.join(seg_save_path, i_class)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

images_save_path = r'D:\NCKU\Thymoma_classfication\data\images2'
if not os.path.exists(images_save_path):
    os.makedirs(images_save_path)

for i_class in classes:
    class_path = os.path.join(images_save_path, i_class)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

images_label_save_path = r'D:\NCKU\Thymoma_classfication\data\images_label2'
if not os.path.exists(images_label_save_path):
    os.makedirs(images_label_save_path)
    
for i_class in classes:
    class_path = os.path.join(images_label_save_path, i_class)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

# In[] Test 1 

# ## forloop of getting patient CT images and segmentation labels
# # data_information = pd.DataFrame(columns = ['Patient_id', 'Label_status'])
# for patient_id in tqdm(patient_list):

    
#     if 'A' not in patient_id:
#         continue
    
#     try:
#         class_type = str(class_info.loc[patient_id].values[0])
#     except:
#         continue
    
#     image_dir = os.path.join(patient_data_dir, patient_id)
    
#     ## check seg label status
#     seg_nrrd_file = glob.glob(image_dir + '/*Segmentation.seg.nrrd')
#     if seg_nrrd_file == []:
#         print('Segmentation label of Patient ' + str(patient_id) + ' is missing')
#         # data_information = data_information.append(pd.DataFrame([[str(patient_id), 'missing']] ,columns = ['Patient_id', 'Label_status']))
#         continue
    
#     else:
#         ### get seg label and align with image
#         seg_label, seg_option = nrrd.read(seg_nrrd_file[0])
        
#         #### judge the shape of seg label
        
#         if len(seg_label.shape) == 4:
#             seg_label_new = seg_label[0] + seg_label[1]
#             seg_label = seg_label_new
        
#         location_raw = seg_option['Segmentation_ReferenceImageExtentOffset'].split(' ')
#         location = np.array([int(location_raw[0]), int(location_raw[1])])
#         label_CT_index = range(int(location_raw[2]), int(location_raw[2]) + seg_label.shape[2])

#     # data_information = data_information.append(pd.DataFrame([[str(patient_id), 'exist']] ,columns = ['Patient_id', 'Label_status']))
#     # data_information.to_csv('data_information.csv', index = False)
    
#     ## get image from nrrd
#     CT_nrrd_file = glob.glob(image_dir + '/*.nrrd')
    
#     for nrrd_file in CT_nrrd_file:
#         if 'Segmentation' in nrrd_file:
#             continue
        
#         print('Find CT images')
#         nrrd_data, _ = nrrd.read(nrrd_file)
#         CT_images = nrrd_data.copy()
    
#     for index_img in range(CT_images.shape[2]):
        
#         if index_img not in label_CT_index:
#             continue
        
#         ### convert label
#         elif index_img in label_CT_index:
            
#             i_image = CT_images[:, :, index_img] ## HU value
#             i_image = rot_and_flip_for_array(i_image) 
#             i_image = convert_soft_tissue_window(i_image) ## convert HU by level and window from mrml
#             i_image = np.uint8(i_image) ## convert to cv2 mode
#             # cv2.imshow('original', i_image)
            
#             ### save original CT images
#             image_name = os.path.basename(image_dir) + 'VS' + str(index_img + 1) + '.png'
#             cv2.imwrite(os.path.join(images_save_path, class_type, image_name), i_image)
            
            
#             i_Seg = seg_label[:, :, label_CT_index.index(index_img)].copy()
            
#             label = np.zeros(i_image.shape)
#             label[location[0] : location[0] + i_Seg.shape[0], location[1] : location[1] + i_Seg.shape[1]] = i_Seg
#             label = rot_and_flip_for_array(label)
            
#             i_image_3c_original = cv2.cvtColor(i_image, cv2.COLOR_GRAY2BGR)
#             i_image_3c_masked = i_image_3c_original.copy()
#             i_image_3c_masked[label == 1] = [128, 174, 128]
            
#             ### save segmentation label
#             cv2.imwrite(os.path.join(seg_save_path, class_type, image_name), label)
            
#             ### save CT images with label color
#             image_result = cv2.addWeighted(i_image_3c_original, 0.55, i_image_3c_masked, 0.45, 0)
#             image_name = os.path.basename(image_dir) + 'VS' + str(index_img + 1) + '.png'
#             cv2.imwrite(os.path.join(images_label_save_path, class_type, image_name), image_result)
#             # cv2.imshow('labeled', image_result)
        
    

# In[] Test 2

## forloop of getting patient CT images and segmentation labels
# data_information = pd.DataFrame(columns = ['Patient_id', 'Label_status'])
for patient_id in tqdm(patient_list):
    patient_number = int(patient_id[1:])
    
    try:
        class_type = str(class_info.iloc[patient_number].values[0])
    except:
        continue
    
    if class_type  == 'nan':
        print(f'{patient_id} is special tumor')
        continue
    
    
    image_dir = os.path.join(patient_data_dir, patient_id)
    
    ## check seg label status
    seg_nrrd_file = glob.glob(image_dir + '/*Segmentation.seg.nrrd')
    if seg_nrrd_file == []:
        print('Segmentation label of Patient ' + str(patient_id) + ' is missing')
        # data_information = data_information.append(pd.DataFrame([[str(patient_id), 'missing']] ,columns = ['Patient_id', 'Label_status']))
        continue
    
    else:
        ### get seg label and align with image
        seg_label, seg_option = nrrd.read(seg_nrrd_file[0])
        
        #### judge the shape of seg label
        
        if len(seg_label.shape) == 4:
            seg_label_new = seg_label[0] + seg_label[1]
            seg_label = seg_label_new
        
        location_raw = seg_option['Segmentation_ReferenceImageExtentOffset'].split(' ')
        location = np.array([int(location_raw[0]), int(location_raw[1])])
        label_CT_index = range(int(location_raw[2]), int(location_raw[2]) + seg_label.shape[2])

    # data_information = data_information.append(pd.DataFrame([[str(patient_id), 'exist']] ,columns = ['Patient_id', 'Label_status']))
    # data_information.to_csv('data_information.csv', index = False)
    
    ## get image from nrrd
    CT_nrrd_file = glob.glob(image_dir + '/*.nrrd')
    
    for nrrd_file in CT_nrrd_file:
        if 'Segmentation' in nrrd_file:
            continue
        
        print('Find CT images')
        nrrd_data, _ = nrrd.read(nrrd_file)
        CT_images = nrrd_data.copy()
    
    for index_img in range(CT_images.shape[2]):
        
        if index_img not in label_CT_index:
            continue
        
        ### convert label
        elif index_img in label_CT_index:
            
            i_image = CT_images[:, :, index_img] ## HU value
            i_image = rot_and_flip_for_array(i_image) 
            i_image = convert_soft_tissue_window(i_image) ## convert HU by level and window from mrml
            i_image = np.uint8(i_image) ## convert to cv2 mode
            # cv2.imshow('original', i_image)
            
            ### save original CT images
            image_name = os.path.basename(image_dir) + 'VS' + str(index_img + 1) + '.png'
            cv2.imwrite(os.path.join(images_save_path, class_type, image_name), i_image)
            
            
            i_Seg = seg_label[:, :, label_CT_index.index(index_img)].copy()
            
            label = np.zeros(i_image.shape)
            label[location[0] : location[0] + i_Seg.shape[0], location[1] : location[1] + i_Seg.shape[1]] = i_Seg
            label = rot_and_flip_for_array(label)
            
            i_image_3c_original = cv2.cvtColor(i_image, cv2.COLOR_GRAY2BGR)
            i_image_3c_masked = i_image_3c_original.copy()
            i_image_3c_masked[label == 1] = [128, 174, 128]
            
            ### save segmentation label
            cv2.imwrite(os.path.join(seg_save_path, class_type, image_name), label)
            
            ### save CT images with label color
            image_result = cv2.addWeighted(i_image_3c_original, 0.55, i_image_3c_masked, 0.45, 0)
            image_name = os.path.basename(image_dir) + 'VS' + str(index_img + 1) + '.png'
            cv2.imwrite(os.path.join(images_label_save_path, class_type, image_name), image_result)
            # cv2.imshow('labeled', image_result)


