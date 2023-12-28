import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


"""
load labels and predicts from csv files
"""
def load_labels_predicts(csv_file_path):

    # csv_file_path = "daily_assess/daily_raw_data/1112.csv"
    df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf-8')

    labels = []
    predicts = []
    for _, row in df.iterrows():
        label = row["Truth"]
        predict = row["Predict"]

        labels.append(label)
        predicts.append(predict)

    # print(labels)
    # print(predicts)
    return labels, predicts

def load_predict_probs(csv_file_path):

    df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf-8')

    data = []
    for _, row in df.iterrows():
        x0 = row["x0"]
        x1 = row["x1"]
        x2 = row["x2"]
        x3 = row["x3"]
        x4 = row["x4"]
        x5 = row["x5"]
        x6 = row["x6"]
        probs = [x0, x1, x2, x3, x4, x5, x6]
        # print(probs)

        data.append(probs)
    
    return data

def create_datasets(daily_data_root):
    data_paths = [os.path.join(daily_data_root, class_dir) for class_dir in os.listdir(daily_data_root) if os.path.exists(os.path.join(daily_data_root, class_dir))]
    
    X = []
    Y = []
    for data_path in data_paths:

        assert data_path[-4:]==".csv" 

        print(f"Loading: {data_path}")
        labels, predicts = load_labels_predicts(data_path)
        predict_probs = load_predict_probs(data_path)

        X.append(predict_probs)
        Y.append(labels)
        # print(f"X: {len(X)} / Y: {len(Y)}")

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(f"X: {X.shape} / Y: {Y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    print(f"X_train: {X_train.shape} / X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} / y_test: {y_test.shape}")

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    return train_loader, test_loader



if __name__ == '__main__':

    # labels, predicts = load_labels_predicts(csv_file_path = "daily_assess/daily_raw_data/1112.csv")
    # print(f"Initial Acc: {accuracy_score(labels, predicts)}")

    # load_predict_probs(csv_file_path = "daily_assess/daily_raw_data/1112.csv")

    daily_data_root = "daily_assess/daily_raw_data/"
    train_loader, test_loader = create_datasets(daily_data_root=daily_data_root)
    
    for features, labels in train_loader:
        print(features.shape)
        print(labels.shape)
