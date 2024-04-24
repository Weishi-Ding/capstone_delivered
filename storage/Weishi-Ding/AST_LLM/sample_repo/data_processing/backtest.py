from numpy.lib.function_base import cov
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
import tushare as ts
pro = ts.pro_api('278fd5f9937a071ec2d63e9651326bc8761eb75fb560bdddc3fc3f32')

class Backtest:
    ##stock_list is the stock to be trade on each day. It should have the form of {date: [stock index list]}
    def __init__(self, initial_money, agent, test_env):
        self.agent = agent
        self.test_env = test_env
        self.test_env.reset()
        # print(self.test_env.cur_stock_df)
        self.start_date = test_start_date_str   #(self.test_env).cur_stock_df.iloc[0]['trade_date'].astype(int).astype(str)
        self.end_date = test_end_date_str       #(self.test_env).cur_stock_df.iloc[-1]['trade_date'].astype(int).astype(str)
        self.money = initial_money
        self.portfolio = None
        self.portfolio_cost = None
        self.portfolio_value = None  ##prices of each stock in the portfolio
        self.stock_codes = ''

        self.dates = self.test_env.cur_stock_df['trade_date'].astype(int).astype(str).apply(adjust_date_format)
        # self.time_delta = time_delta(adjust_date_format(self.start_date), adjust_date_format(self.end_date)) ##use .date to extract the date difference
        self.cur_date = 0 ##serves as a counter, increment by 1 for each date. Maximum is date difference
        # self.stock_list = stock_list
        self.total_stock_traded = 0
        self.num_win_stock = 0
        self.daily_returns = []
        self.portfolio_trace = []
        self.benchmark_stock = '000300.SH' ##沪深300
        self.benchmark_df = None
        self.risk_free_rate = np.mean(np.array([3.203, 2.789, 2.879, 2.678])/100)
        self.win_stock_traded = []
        self.loss_stock_traded = []


        ###metrics of evaluation
        self.total_returns = []   ##cumulative retuns of daily returns
        self.anual_return = 0
        self.beta = 0
        self.alpha = 0
        self.benchmark_returns = [] ##基准收益率
        self.sharpe_ratio = 0
        self.win_rate = 0
        self.max_drawdown = 0
        self.information_ratio = 0
        self.return_volatility = 0
        self.profit_loss_ratio = 0



    def execute(self):
        self.agent.policy_net.to(device)
        self.agent.target_net.to(device)
        state = self.test_env.reset() ##get the start state
        self.agent.portfolio = 0
        state = np.append(state, self.agent.portfolio)
        terminated = False
        ##set a window to calculate win rate
        time_step = 0
        buy_time = 0
        buy_time_fixed = False
        sell_time = 0

        ##set the policy net to train mode
        self.agent.policy_net.eval()
        self.agent.target_net.eval()

        with torch.no_grad():
          while not terminated:
              self.portfolio_trace.append(self.agent.portfolio)
              res = self.agent.policy_net(torch.tensor(state).to(device).float())
              _, indices = torch.max(res, 0)
              action = action_space[indices.item()]
              next_state, reward, terminated, new_portfolio = self.test_env.step(action, self.agent.portfolio)
              # reward = reward * cash_proportion
              ##check if the action is valid, and then mark the buy/sell time
              if action > 0 and self.agent.portfolio != new_portfolio and (not buy_time_fixed):
                  buy_time = time_step
                  buy_time_fixed = True
              if action < 0 and self.agent.portfolio != new_portfolio: sell_time = time_step
              if terminated:
                  ##ensure that the experience for termination is not added to the buffer
                  break
              next_state = np.append(next_state, new_portfolio)
              self.agent.portfolio = new_portfolio
              # self.agent.exp_buffer.add(Experience(state, action, reward, next_state))
              state = next_state
              if reward >= 100:
                  reward =10
              if reward <= -100:
                  reward = -10
              self.daily_returns.append(reward)
              time_step += 1
              ##After buy and hold a stock, check if the total return is positive
              if sell_time > buy_time:
                  self.total_stock_traded += 1
                  temp = self.daily_returns[buy_time:sell_time]
                  res = (1 + np.array(temp)/100).cumprod()-1

                  if res[-1] > 0:
                      self.win_stock_traded.append(res[-1])
                      self.num_win_stock += 1
                  else:
                      self.loss_stock_traded.append(-res[-1])
                  ##reset the buy and sell time
                  buy_time = 0
                  buy_time_fixed = False
                  sell_time = 0

    def cal_total_return(self):
        self.total_returns = (1 + np.array(self.daily_returns)/100).cumprod() - 1
        return self.total_returns
    
    def cal_annualized_return(self):
        self.annualized_return = (self.total_returns[-1] + 1)**(365/len(self.daily_returns)) - 1
        return self.annualized_return

    def cal_benchmark(self):
        self.benchmark_df = pro.index_daily(ts_code=self.benchmark_stock, start_date=self.start_date, end_date=self.end_date)
        self.benchmark_df = self.benchmark_df.sort_values(by='trade_date', ascending=True)
        # df = df[df['trade_date'].isin(self.test_env.cur_stock_df['trade_date'].astype(str))]
        self.benchmark_returns = self.benchmark_df['pct_chg']
        self.benchmark_returns = ((1 + np.array(self.benchmark_returns)/100).cumprod() - 1)
        return self.benchmark_returns

    def cal_information_ratio(self):
        df = self.benchmark_df
        df = df[df['trade_date'].isin(self.test_env.cur_stock_df['trade_date'].astype(str))]
        ##ensure the length are equal
        track_error = np.array(self.daily_returns) - np.array(df.iloc[1:-1]['pct_chg'])
        self.information_ratio = (self.total_returns[-1] - self.benchmark_returns[-1])/np.std(track_error, ddof=1)
        return self.information_ratio

    def cal_beta(self):
        cov = np.cov(np.array(self.daily_returns), np.array(self.benchmark_returns[1:len(self.daily_returns)+1]))
        self.beta = cov[0][1]/np.var(self.benchmark_returns[1:len(self.daily_returns)+1])
        return self.beta

    def cal_alpha(self):
        self.alpha = self.total_returns[-1] - (self.risk_free_rate + self.beta*(self.benchmark_returns[-1] - self.risk_free_rate))
        return self.alpha

    def cal_sharpe_ratio(self):
        self.sharpe_ratio = (self.total_returns[-1] - self.risk_free_rate)/np.std(self.daily_returns, ddof=1)
        return self.sharpe_ratio

    def cal_win_rate(self):
        if self.total_stock_traded == 0:
            self.win_rate = 1 if self.total_returns[-1] > 0 else 0
            return self.win_rate
        self.win_rate = float(self.num_win_stock/self.total_stock_traded)
        return self.win_rate

    def cal_profit_loss_ratio(self):
        if self.loss_stock_traded == []:
            self.profit_loss_ratio = np.mean(self.win_stock_traded)
            return self.profit_loss_ratio
        self.profit_loss_ratio = np.sum(self.win_stock_traded)/np.sum(self.loss_stock_traded)
        return self.profit_loss_ratio

    def cal_return_volatility(self):
        self.return_volatility = np.std(self.daily_returns, ddof=1)
        return self.return_volatility

    def calculate_max_drawdown(self):
        """
        step1: Calculate the cumulative returns
        step2: Calculate the running maximum
        step3: Calculate drawdowns as the difference between the running max and the cumulative returns
        step4: max
        """
        cumulative_returns = (1 + np.array(self.daily_returns)/100).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        self.max_drawdown = drawdowns.max()
        return self.max_drawdown

    def evaluate(self):
        self.cal_benchmark()
        self.cal_total_return()
        self.cal_annualized_return()
        self.cal_sharpe_ratio()
        self.cal_win_rate()
        self.cal_profit_loss_ratio()
        self.cal_return_volatility()
        self.calculate_max_drawdown()
        self.cal_information_ratio()
        self.cal_beta()
        self.cal_alpha()
        return np.array([self.total_returns[-1], self.sharpe_ratio, self.win_rate, self.profit_loss_ratio, self.return_volatility, self.max_drawdown, self.information_ratio, self.beta, self.alpha, self.annualized_return])


    def show_result(self):
        self.evaluate()

        # print(self.benchmark_returns)
        # Sample Data
        # Assuming `daily_returns` and `benchmark_returns` are lists of your daily return values
        # and `dates` is a list of corresponding date values.

        # self.benchmark_df = pro.index_daily(ts_code=self.benchmark_stock, start_date=self.start_date, end_date=self.end_date)
        # self.benchmark_df = self.benchmark_df.sort_values(by='trade_date', ascending=True)
        # # df = df[df['trade_date'].isin(self.test_env.cur_stock_df['trade_date'].astype(str))]
        # self.benchmark_returns = self.benchmark_df['pct_chg']
        # self.benchmark_returns = ((1 + np.array(self.benchmark_returns)/100).cumprod() - 1)

        stock_df = pro.daily(ts_code=self.test_env.cur_stock_code, start_date=self.start_date, end_date=self.end_date)
        stock_df = stock_df.sort_values(by='trade_date', ascending=True)
        stock_returns = stock_df['pct_chg']
        stock_returns = ((1 + stock_returns/100).cumprod() -1)

        # Create figure
        fig = go.Figure()
        # fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        #             vertical_spacing=0.02,
        #             subplot_titles=('Returns', 'Cash Pool Usage'),
        #             row_heights=[0.7, 0.3])
        # Add traces
        fig.add_trace(go.Scatter(x=self.dates, y=self.total_returns,  mode='lines', name=f'Strategy ({self.test_env.cur_stock_code})',
                                fill='tozeroy', fillcolor='rgba(0,100,100,0.2)'))
        fig.add_trace(go.Scatter(x=self.dates, y=self.benchmark_returns, mode='lines', name=f'Benchmark ({self.benchmark_stock})',
                                fill='tozeroy', fillcolor='rgba(200,0,80,0.2)'))
        fig.add_trace(go.Scatter(x=self.dates, y=stock_returns, mode='lines', name=f'Stock Price ({self.test_env.cur_stock_code})',
                                fill='tozeroy', fillcolor='rgba(0,200,80,0.2)'))
        # fig.add_trace(go.Scatter(x=dates, y=self.portfolio_trace, name='Cash Pool Usage'), row=2, col=1)

        fig.add_annotation(x=0, y=1.25, xref="paper", yref="paper",
                          text=
                          f"<span style='font-size: 12px; color: grey;'>Total Return:</span> <b>{self.total_returns[-1]*100:.2f}%</b>        " +
                          f"<span style='font-size: 12px; color: grey;'>Annual Return:</span> <b>{self.annualized_return*100:.2f}%</b>        " +
                          f"<span style='font-size: 12px; color: grey;'>Benchmark Return:</span> <b>{self.benchmark_returns[-1]*100:.2f}%</b>        " +
                          f"<span style='font-size: 12px; color: grey;'>Alpha:</span> <b>{self.alpha:.2f}</b>        " +
                          f"<span style='font-size: 12px; color: grey;'>Beta:</span> <b>{self.beta:.2f}</b>        " +
                          f"<span style='font-size: 12px; color: grey;'>Sharpe Ratio:</span> <b>{self.sharpe_ratio:.2f}</b>",
                          showarrow=False, font=dict(size=15), borderwidth=2, borderpad=4)

        fig.add_annotation(x=0, y=1.15, xref="paper", yref="paper",
                          text=
                          f"<span style='font-size: 12px; color: grey;'>Win Rate:</span> <b>{self.cal_win_rate() * 100:.2f}%\n</b>         " +
                          f"<span style='font-size: 12px; color: grey;'>Max Drawdown:</span> <b>{self.calculate_max_drawdown() * 100:.2f}%</b>         " +
                          f"<span style='font-size: 12px; color: grey;'>Information Ratio:</span> <b>{self.information_ratio:.2f}</b>         " +
                          f"<span style='font-size: 12px; color: grey;'>Return Volatility:</span> <b>{self.cal_return_volatility():.2f}</b>         " +
                          f"<span style='font-size: 12px; color: grey;'>Profit Loss Ratio:</span> <b>{self.cal_profit_loss_ratio():.2f}</b>" ,
                          showarrow=False, font=dict(size=15), borderwidth=2, borderpad=4)

        # Update layout options
        fig.update_layout(
            title='Strategy Backtest',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            xaxis_rangeslider_visible=True,
            yaxis_tickformat=',.0%',  # Display y-axis ticks as percentages
            title_y=0.98,  # Adjust title position to make space for the banner
            title_x=0.05,
            legend=dict(
                y=-0.8,  # Adjusts vertical position. Values below 0 move the legend below the chart.
                x=0.5,  # Adjusts horizontal position. 0.5 centers the legend.
                xanchor='center',  # Ensures the x position refers to the center of the legend.
                orientation='h'  # Makes the legend horizontal. Remove if you prefer a vertical legend.
            ),
            margin=dict(t=100),
            # margin=dict(l=50, r=50, t=100, b=50)
            # height=600
        )

        # Show figure
        fig.show()