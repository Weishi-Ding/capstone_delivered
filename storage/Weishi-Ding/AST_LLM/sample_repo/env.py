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

class Environment():
    def __init__(self, df):
        self.df = df
        self.stock_list = df['ts_code'].unique()
        self.cur_stock_code = ''   ##the stock chosen for the current episode
        self.cur_stock_df = None
        self.time_step = 0

    def reset(self):
        '''
        reset the environment by choosing a new stock
        steps:
          1. choose a stock from the stock list
          2. set the cur_stock_df to be the dataframe or the chosen stock
          3. sort the cur_stdck_df by trade_date
          4. reset the index of cur_stock_df
        '''

        self.stock_index = np.random.randint(0, len(self.stock_list))
        self.cur_stock_code = self.stock_list[self.stock_index]
        self.cur_stock_df = self.df[self.df['ts_code'] == self.cur_stock_code]
        self.cur_stock_df = self.cur_stock_df.sort_values(by=['trade_date'])
        self.cur_stock_df = self.cur_stock_df.reset_index(drop=True)
        self.cur_stock_df = self.cur_stock_df.iloc[:, 1:]
        ##reset time_step
        self.time_step = 0

        return self.cur_stock_df.drop(['trade_date', 'pct_change'], axis = 1).iloc[0,:].values

    def step(self, action, portfolio):
        '''
        steps:
          1. get the next state
          2. get the reward
        '''
        terminated = False
        reward = 0
        new_portfolio = 0
        factor = 0
        # if self.cur_stock_df.loc[self.time_step, 'trend'] != 1:
        #     factor = downtrend_factor
        # else:
        #     factor = uptrend_factor

        # yest_close = float(self.cur_stock_df.loc[ self.time_step, 'close']) ##yesterday's closing price
        # if yest_close == 0.0: yest_close = 1.0
        

        self.time_step += 1
        if self.time_step >= len(self.cur_stock_df) - 1:
            terminated = True
            return None, reward, terminated, None

        next_state = self.cur_stock_df.drop(['trade_date', 'pct_change'], axis = 1).iloc[self.time_step,:].values

        ##if action is to hold, then the reward depends on current portfolio
        # if (action == 0):
        #     new_portfolio = portfolio ##portfolio is unchanged
        #     reward = self.cur_stock_df.loc[self.time_step, 'pct_change'] * new_portfolio
        # ##if the action is to buy
        # if (action > 0):
        #     new_portfolio = min(action + portfolio, 1)
        #     reward = self.cur_stock_df.loc[self.time_step, 'pct_change'] * portfolio + \
        #              self.cur_stock_df.loc[self.time_step, 'pct_change'] * (new_portfolio - portfolio) - cost_rate * (new_portfolio - portfolio)
        # ##if the action is to sell, the return is the pct_change between yesterday's close and today's open
        # if (action < 0):
        #     today_open = float(self.cur_stock_df.loc[self.time_step, 'open'])
        #     pct_change = (today_open - yest_close) / yest_close * 100
        #     reward = portfolio * (pct_change - stamp_duty_rate * 10)
        #     new_portfolio = max(action + portfolio, 0)
        if (action == 0):
            new_portfolio = portfolio ##portfolio is unchanged
            reward = self.cur_stock_df.loc[self.time_step, 'pct_change'] * new_portfolio
        ##if the action is to buy
        if (action > 0):
            new_portfolio = min(action + portfolio, 1)
            reward = self.cur_stock_df.loc[self.time_step, 'pct_change'] * new_portfolio
        ##if the action is to sell
        if (action < 0):
            new_portfolio = max(action + portfolio, 0)
            reward = -self.cur_stock_df.loc[self.time_step, 'pct_change'] * portfolio

        if reward < -5.0: reward -= 100 ##big penalty for huge drawdown
        if reward > 5.0: reward += 100 ##big reward for huge profit

        
        return next_state, reward, terminated, new_portfolio



