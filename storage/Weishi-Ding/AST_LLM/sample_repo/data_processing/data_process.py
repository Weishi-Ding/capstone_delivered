import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.utils.data import Dataset, DataLoader
from torch.optim import optimizer
from collections import namedtuple
from utils import ndarray_to_tensor, oneDarray_to_tensor, extract_tensors, get_cur_Qs, get_target_Qs, adjust_date_format

def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

def add_avg(df):
    df['avg'] = df['close'].rolling(window=5).mean()
    df = df.dropna()
    return df

def process_data(df, test=False):
    df = df.dropna()
    df = df.reset_index(drop=True)
    # df = df.iloc[:, 1:]
    if not test:
      df['pct_change'] = df['pct_change'].clip(-10,10)
    temp = df.drop(['trade_date', 'pct_change'], axis = 1)
    # temp = temp.groupby('ts_code').apply(add_avg)
    # temp = temp.reset_index(drop=True)
    temp = temp.groupby('ts_code').transform(normalize_column)
    # temp['trend'] = temp['close'] > temp['avg']
    # temp['trend'] = temp['trend'].astype(int)
    # temp = temp.drop(['avg'], axis=1)
    df[temp.columns] = temp
    df = df.dropna()
    return df