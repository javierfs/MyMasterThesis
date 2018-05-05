# Data analysis packages:
import pandas as pd
import numpy as np
# from datetime import datetime as dt

# Visualization packages:
import seaborn as sns
import matplotlib.pyplot as plt

filename = 'TotalDataBase.pkl'
df_total = pd.read_pickle(filename)


filename = 'ClevelandHeartDisease_raw.pkl'
df_clev = pd.read_pickle(filename)

filename = 'HungarianHeartDisease_raw.pkl'
df_hung = pd.read_pickle(filename)


filename = 'SwitzerlandHeartDisease_raw.pkl'
df_sw = pd.read_pickle(filename)

filename = 'VaLongBeachHeartDisease_raw.pkl'
df_longBeach = pd.read_pickle(filename)




# For simplicity: replace values 1,2,3,4 of Heart Disease -> 1, Healthy -> 0
for i in range(1, 5):
    df_total['heartdisease'] = df_total['heartdisease'].replace(i, 1)
    df_clev['heartdisease'] = df_clev['heartdisease'].replace(i, 1)
    df_hung['heartdisease'] = df_hung['heartdisease'].replace(i, 1)
    df_sw['heartdisease'] = df_sw['heartdisease'].replace(i, 1)
    df_longBeach['heartdisease'] = df_longBeach['heartdisease'].replace(i, 1)


param2int = ['age','sex','cp','trestbps','chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal', 'heartdisease']
param2float = ['oldpeak']

list_db = [df_clev, df_hung, df_sw, df_longBeach]

for db in list_db:
    for key,val in db.items():
        if key == 'chol':
            if isinstance(key, str):
                db['chol'] = db['chol'].replace('0', '-9')
            else:
                db['chol'] = db['chol'].replace(0, -9)

        db['ca'] = db['ca'].replace(9, -9)

        if isinstance(key, str):
            db[key] = db[key].replace('?', '-9')
    for param, value in db.items():
        for val in param2int:
            if val in db:
                db[val] = db[val].apply(lambda x: int(float(x)));
        for val in param2float:
            if val in db:
                db[val] = db[val].apply(lambda x: float(x));
    db['sex'] = pd.Categorical(db.sex)
    db['fbs'] = pd.Categorical(db.fbs)
    db['cp'] = pd.Categorical(db.cp)
    db['exang'] = pd.Categorical(db.exang)
    db['thal'] = pd.Categorical(db.thal)
    db['restecg'] = pd.Categorical(db.restecg)
    db['slope'] = pd.Categorical(db.slope)
    db['ca'] = pd.Categorical(db.ca)
    db['heartdisease'] = pd.Categorical(db .heartdisease)


    for key,val in df_total.items():
        if key == 'chol':
            df_total[key] = df_total[key].replace(0, -9)
        if key == 'ca':
            df_total[key] = df_total[key].replace(9, -9)

        if isinstance(key, str):
            df_total[key] = df_total[key].replace('?', '-9')
    for param, value in df_total.items():
        for val in param2int:
            if val in df_total:
                df_total[val] = df_total[val].apply(lambda x: int(float(x)));
        for val in param2float:
            if val in df_total:
                df_total[val] = df_total[val].apply(lambda x: float(x));

    df_total['sex'] = pd.Categorical(df_total.sex)
    df_total['fbs'] = pd.Categorical(df_total.fbs)
    df_total['cp'] = pd.Categorical(df_total.cp)
    df_total['exang'] = pd.Categorical(df_total.exang)
    df_total['thal'] = pd.Categorical(df_total.thal)
    df_total['restecg'] = pd.Categorical(df_total.restecg)
    df_total['slope'] = pd.Categorical(df_total.slope)
    df_total['ca'] = pd.Categorical(df_total.ca)
    df_total['heartdisease'] = pd.Categorical(df_total.heartdisease)






filename = 'TotalDataBase.pkl'
df_total.to_pickle(filename)

filename = 'ClevelandHeartDisease.pkl'
list_db[0].to_pickle(filename)

filename = 'HungarianHeartDisease.pkl'
list_db[1].to_pickle(filename)

filename = 'SwitzerlandHeartDisease.pkl'
list_db[2].to_pickle(filename)

filename = 'VaLongBeachHeartDisease.pkl'
list_db[3].to_pickle(filename)


print('hola')