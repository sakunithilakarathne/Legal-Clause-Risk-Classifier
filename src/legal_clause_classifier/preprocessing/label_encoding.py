import pandas as pd
import numpy as np
import pickle
import os
import json

from sklearn.preprocessing import MultiLabelBinarizer


def multi_hot_encoding(train_df, val_df, test_df):
    with open("artifacts/label_list.json") as f:
        all_labels = json.load(f)

    mlb = MultiLabelBinarizer(classes=all_labels)
    y_train = mlb.fit_transform(train_df["categories_list"])
    y_val = mlb.transform(val_df["categories_list"])
    y_test = mlb.transform(test_df["categories_list"])

    
    