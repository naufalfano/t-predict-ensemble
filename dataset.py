import pandas as pd
import numpy as np

def get_data():
    path_file = './dataset/bank-marketing/bank-additional-full.csv'
    df = pd.read_csv(path_file, delimiter=';')

    return df