import numpy as np
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor
import pandas as pd
import os


def load_training_data():
    train = pd.read_csv("data/train_02032024.csv")