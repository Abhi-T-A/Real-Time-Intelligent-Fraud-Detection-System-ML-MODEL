import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)