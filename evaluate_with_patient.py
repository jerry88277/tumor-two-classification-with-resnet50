# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:31:35 2021

@author: Jerry
"""

import os
import re
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
# In[]

def get_patient_ID(patient_slice_ID):
    
    patient_ID = re.split('\\\\|.png', patient_slice_ID)
    patient_ID = patient_ID[-2][:4]
    
    return patient_ID
# In[]
save_path = r'D:\NCKU\Thymoma_classfication\prediction\2C_new'
if not os.path.exists(save_path):
    os.makedirs(save_path)

csv_dir = r'D:\NCKU\Thymoma_classfication\prediction\2C'

csv_list = os.listdir(csv_dir)

for i_csv in csv_list:
    i_csv_path = os.path.join(csv_dir, i_csv)
    prediction = pd.read_csv(i_csv_path, names = ['patient_slice_ID', 'pred', 'true'])
    prediction.drop(index = 0, inplace = True)
    prediction.reset_index(drop = True, inplace = True)
    
    prediction['patient_ID'] = prediction['patient_slice_ID'].apply(get_patient_ID)
    
    unique_ID = np.unique(prediction['patient_ID'])
    
    new_prediction_result = pd.DataFrame(columns = ['patient_ID', 'pred', 'true'])
    
    for ID in unique_ID:
        condiction = prediction['patient_ID'] == ID
        pred = prediction['pred'][condiction]
        true = prediction['true'][condiction]
        
        pred_by_most = np.argmax(np.bincount(pred))
        true = np.argmax(np.bincount(true))

        temp_result = pd.DataFrame([[ID, pred_by_most, true]], columns = ['patient_ID', 'pred', 'true'])
        new_prediction_result = new_prediction_result.append(temp_result)
        
    new_prediction_result.reset_index(drop=True)
    new_prediction_result.to_csv(f'{os.path.join(save_path, i_csv)}', index = False)
    
# In[]
SEED_list = range(4019, 4019 + 10)

save_path = r'D:\NCKU\Thymoma_classfication\prediction\2C_metrics_new'
if not os.path.exists(save_path):
    os.makedirs(save_path)

csv_dir = r'D:\NCKU\Thymoma_classfication\prediction\2C_new'

csv_list = os.listdir(csv_dir)

metrice_result = pd.DataFrame(columns = ['Seed','Accuracy', 'Precision', 'Recall', 'F1 score'])

for index, i_csv in enumerate(csv_list):
    i_csv_path = os.path.join(csv_dir, i_csv)
    prediction = pd.read_csv(i_csv_path)
    
    y_test = prediction.true
    y_pred = prediction.pred
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    F1_score = f1_score(y_test, y_pred, average='weighted')

    temp_result = pd.DataFrame([[SEED_list[index], accuracy, precision, recall, F1_score]] ,columns = ['Seed', 'Accuracy', 'Precision', 'Recall', 'F1 score'])
    metrice_result = metrice_result.append(temp_result)

metrice_result.reset_index(drop=True)
metrice_result.to_csv(f'{os.path.join(save_path, "new_metrice_result.csv")}', index = False)









