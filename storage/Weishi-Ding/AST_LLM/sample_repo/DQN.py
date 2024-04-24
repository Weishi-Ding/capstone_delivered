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

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim))
        # self.target_net = self.policy_net.copy()

    def forward(self, x):
        return self.model(x)


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))


class Experience_Buffer():
    def __init__(self, buffer_cap=1000):
        self.buffer = []
        self.buffer_size = 0
        self.buffer_cap = buffer_cap

    def add(self, experience : Experience):
        if self.buffer_size + 1 >= self.buffer_cap:
            self.buffer.pop(0)

        self.buffer.append(experience)
        self.buffer_size += 1

    def sample(self, batch_size):
        np.random.shuffle(self.buffer)
        return self.buffer[:batch_size]

    def can_provide_sample(self, batch_size):
        return self.buffer_size >= batch_size
