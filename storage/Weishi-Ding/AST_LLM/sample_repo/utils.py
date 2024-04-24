import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.utils.data import Dataset, DataLoader
from torch.optim import optimizer
from collections import namedtuple
# from utils import ndarray_to_tensor, oneDarray_to_tensor, extract_tensors, get_cur_Qs, get_target_Qs, adjust_date_format

def ndarray_to_tensor(array, batch_size, num_features):
    res = list(map(lambda x: torch.tensor(x, dtype=torch.float32), array))
    res = torch.cat(res)
    res = res.view(batch_size, num_features)
    return res

def oneDarray_to_tensor(array, batch_size):
    res = torch.tensor(array, dtype=torch.float32).view(batch_size, -1)
    return res


def extract_tensors(experiences, batch_size, num_features):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = ndarray_to_tensor(batch.state, batch_size, len(batch.state[0]))
    t2 = oneDarray_to_tensor(batch.action, batch_size)
    t3 = oneDarray_to_tensor(batch.reward, batch_size)
    t4 = ndarray_to_tensor(batch.next_state, batch_size, len(batch.next_state[0]))

    return (t1,t2,t3,t4)

def get_cur_Qs(policy_net, states, actions):
    res = policy_net(states)
    res = res.gather(1, actions.to(torch.int64))
    return res

def get_target_Qs(target_net, next_states):
    res = target_net(next_states)
    val, indices = torch.max(res, 1)
    val = val.unsqueeze(1)
    return val

def adjust_date_format(date_str):
    # Convert the date strings to datetime objects YYYY-MM-DD
    return date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:]







