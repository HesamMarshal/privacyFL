import os
import pandas as pd
import numpy as np
import config
from utils.model_evaluator import ModelEvaluator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

file_path = os.path.join(os.getcwd(), 'datasets', 'kddcup.csv')

dataset = pd.read_csv(file_path)
print(f"Dataset: KDD CUP")
print(
    f"Dataset Rows: {dataset.shape[0]}\nDataset Cols: {dataset.shape[1]} ")
print(f"Classifier: {config.CLASSIFIER}")

x = dataset.drop(dataset.iloc[:, 41:42], axis=1)
x = x.drop(x.iloc[:, 1:4], axis=1)
x = np.array(x)
y = dataset.iloc[:, 41].apply(lambda x: 1 if x == 'normal.' else 0)

y = np.array(y)
scaler = MinMaxScaler()
scaler.fit(x)
X = scaler.transform(x)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
