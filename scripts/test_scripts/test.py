
import numpy as np
import torch 
import scipy.io as sio
import pandas as pd

df = pd.read_csv('dataset/annotation.csv')
print(df.values)
print(df.columns)
locations = ["a", "b", "c", "d", "e"]
loc_cols = [f"user_{k}_location" for k in range(1, 6)]
for j in range(1, 6):
    df_count_j = df[df["number_of_users"] == j]
    print(f"Count {j}: {len(df_count_j)}")
    for i in locations:
        mask = df_count_j[loc_cols].eq(i).any(axis=1)
        df_location = df_count_j[mask]
        print(f"Location {i}: {len(df_location)}")       