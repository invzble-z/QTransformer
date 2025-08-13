import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# Read the CSV file
# dataset_path = '/home/u1/Desktop/Gra_pr/QTransformer/dataset/Dataco_dataset/example_DataCoSupplyChainDataset.csv'
# dataset_path = '/home/u1/Desktop/Gra_pr/QTransformer/dataset/Dataco_dataset/DataCoSupplyChainDataset.csv'
dataset_path = '/home/u1/Desktop/Gra_pr/QTransformer/dataset/Dataco_dataset/70k_DataCoSupplyChainDataset.csv'
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')


print(df['Order Region'].unique())
