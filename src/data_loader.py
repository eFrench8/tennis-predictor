import pandas as pd

def load_data(path='data/atp_tennis.csv'):
  return pd.read_csv(path)