import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_data():
    path_file = './dataset/bank-marketing/bank-additional-full.csv'
    df = pd.read_csv(path_file, delimiter=';')

    for feature in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']:
        df[feature] = LabelEncoder().fit_transform(df[feature])

    scaler = StandardScaler()
    for feature in ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']:
        df[feature] = scaler.fit_transform(df[[feature]])

    return df