import glob
import json
import os
import pandas as pd
import math
import numpy as np
import sklearn
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import Imputer



Hungarian_data_folder = 'dataset/reprocessed.hungarian.data.txt.csv'
Cleveland_data_folder = 'dataset/processed.cleveland.data.txt'
sw_data_folder = 'dataset/processed.switzerland.data.txt'
va_data_folder = 'dataset/processed.va.data.txt'


np.set_printoptions(threshold=np.nan) #see a whole array when we output it

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']

ClevelandHeartDisease = pd.read_csv(Cleveland_data_folder, names = names) #gets Cleveland data
HungarianHeartDisease = pd.read_csv(Hungarian_data_folder, names = names) #gets Hungary data
SwitzerlandHeartDisease = pd.read_csv(sw_data_folder, names = names) #gets Switzerland data
VaLongBeachHeartDisease = pd.read_csv(va_data_folder, names = names) #gets Switzerland data

df_clev = ClevelandHeartDisease
df_hung = HungarianHeartDisease
df_sw = SwitzerlandHeartDisease
df_longBeach = VaLongBeachHeartDisease










filename = 'ClevelandHeartDisease_raw.pkl'
ClevelandHeartDisease.to_pickle(filename)

filename = 'HungarianHeartDisease_raw.pkl'
HungarianHeartDisease.to_pickle(filename)

filename = 'SwitzerlandHeartDisease_raw.pkl'
SwitzerlandHeartDisease.to_pickle(filename)

filename = 'VaLongBeachHeartDisease_raw.pkl'
VaLongBeachHeartDisease.to_pickle(filename)




TotalDatabase = [ClevelandHeartDisease, HungarianHeartDisease, SwitzerlandHeartDisease, VaLongBeachHeartDisease] #combines all arrays into a list

df_TotalDatabase = pd.concat(TotalDatabase, ignore_index=True)  # 1st row as the column name
filename = 'TotalDataBase.pkl'
df_TotalDatabase.to_pickle(filename)



