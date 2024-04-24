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
from agent import Agent
from env import Environment
from backtest import Backtest

class Portfolio():
    def __init__(self, stock_list):
        self.stock_list = stock_list
        self.agent_list = [Agent(model) for i in range(len(stock_list))]
        self.backtests = []
        self.cash_allocation = np.ones(len(stock_list)) / len(stock_list)  #evenly divided initially
        self.reallocate_period = 3 ##reallocate cash after 10 days
        self.retrain_period = 30
        self.shared_buffer = Experience_Buffer(500)
        self.batch_size = 32
        # self.time_step = 0

    def initialize(self):
        for i, stock_code in enumerate(self.stock_list):
            df = pro.stk_factor(ts_code=stock_code, start_date=train_start_date_str, end_date=test_end_date_str,
                                fields='ts_code,trade_date,close,open,high,low,pct_change,vol,macd_dif,\
                                        macd_dea,macd,kdj_k,kdj_d,kdj_j,rsi_6,rsi_12,rsi_24,boll_upper,boll_mid,boll_lower,cci')
            df = process_data(df)
            train_df = df[(df['trade_date'] >= train_start_date_str) & (df['trade_date'] <= train_end_date_str)]
            further_train_env = Environment(train_df)
            self.agent_list[i].train(further_train_env, val_env, 3, batch_size, epsilon, gamma, lr, device)
            test_df = pro.stk_factor(ts_code=stock_code, start_date=test_start_date_str, end_date=test_end_date_str,
                                fields='ts_code,trade_date,close,open,high,low,pct_change,vol,macd_dif,\
                                        macd_dea,macd,kdj_k,kdj_d,kdj_j,rsi_6,rsi_12,rsi_24,boll_upper,boll_mid,boll_lower,cci')
            test_df = df[df['trade_date'] >= test_start_date_str]
            test_env = Environment(test_df)
            self.backtests.append(Backtest(0, self.agent_list[i], test_env))

    def reallocate_cash(self):
        performances = []
        for backtest in self.backtests:
            temp = backtest.daily_returns[-self.reallocate_period:]
            temp_performance = np.array(temp).cumprod()[-1]
            performances.append(temp_performance)
            idx_rank = np.array(performances).argsort()
            cash_weights = np.array([1/math.log(i+2) for i in idx_rank])
            self.cash_allocation = cash_weights / cash_weights.sum()
        # print(self.cash_allocation)

    def retrain(self):
        loss_fn = nn.MSELoss()

        if self.shared_buffer.can_provide_sample(self.batch_size):
            for backtest in self.backtests:
                backtest.agent.policy_net.train()
                backtest.agent.target_net.train()
                optimizer = torch.optim.Adam(backtest.agent.policy_net.parameters(), lr=lr)

                experiences = self.shared_buffer.sample(self.batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences, self.batch_size, input_dim)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                current_q_values = get_cur_Qs(backtest.agent.policy_net, states, actions)
                target_q_values = get_target_Qs(backtest.agent.target_net, next_states)
                target_q_values = (target_q_values * gamma) + rewards

                loss = loss_fn(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                backtest.agent.policy_net.eval()
                backtest.agent.target_net.eval()

    def run(self):
        states_list = []    ##store state for each backtest
        terminated_list = [False] * len(self.backtests)
        buy_time = [0] * len(self.backtests)
        buy_time_fixed = [False] * len(self.backtests)
        sell_time = [0] * len(self.backtests)
        time_step = 0

        for b in self.backtests:
            b.agent.policy_net.to(device)
            b.agent.target_net.to(device)
            b.agent.policy_net.eval()
            b.agent.target_net.eval()

            state = b.test_env.reset() ##get the start state
            b.agent.portfolio = 0
            state = np.append(state, b.agent.portfolio)
            states_list.append(state)
            b.test_env.reset()

        ##only terminated when all agent is terminated
        with torch.no_grad():
            while sum(terminated_list) != len(terminated_list):
                ##Reallocate the cash if needed
                if time_step > 0 and time_step % self.reallocate_period == 0:
                    self.reallocate_cash()
                if time_step > 0 and time_step % self.retrain_period == 0:
                    torch.set_grad_enabled(True)
                    self.retrain()
                    # Disable gradients again
                    torch.set_grad_enabled(False)

                ##normal case
                for i, b in enumerate(self.backtests):
                    b.portfolio_trace.append(b.agent.portfolio)
                    res = b.agent.policy_net(torch.tensor(states_list[i]).to(device).float())
                    _, indices = torch.max(res, 0)
                    action = action_space[indices.item()]
                    next_state, reward, terminated, new_portfolio = b.test_env.step(action, b.agent.portfolio)
                    next_state = np.append(next_state, new_portfolio)

                    if action > 0 and b.agent.portfolio != new_portfolio and (not buy_time_fixed[i]):
                        buy_time[i] = time_step
                        buy_time_fixed[i] = True
                    if action < 0 and b.agent.portfolio != new_portfolio:
                        sell_time[i] = time_step
                    terminated_list[i] = terminated
                    if terminated:
                        ##ensure that the experience for termination is not added to the buffer
                        continue

                    if reward >= 100:
                        self.shared_buffer.add(Experience(states_list[i], action, reward, next_state))
                        reward -= 100
                    if reward <= -100:
                        self.shared_buffer.add(Experience(states_list[i], action, reward, next_state))
                        reward += 100
                    b.daily_returns.append(reward * self.cash_allocation[i])
                    ##update states_list
                    states_list[i] = next_state
                    ##update portfolio
                    b.agent.portfolio = new_portfolio

                    ##After buy and hold a stock, check if the total return is positive
                    if sell_time[i] > buy_time[i]:
                        b.total_stock_traded += 1
                        temp = b.daily_returns[buy_time[i]:sell_time[i]]
                        res = (1 + np.array(temp)/100).cumprod()-1

                        if res[-1] > 0:
                            b.win_stock_traded.append(res[-1])
                            b.num_win_stock += 1
                        else:
                            b.loss_stock_traded.append(-res[-1])
                        ##reset the buy and sell time
                        buy_time[i] = 0
                        buy_time_fixed[i] = False
                        sell_time[i] = 0
                time_step += 1

    def evaluate(self):
        mean_metrics =  self.backtests[0].evaluate()
        mean_total_returns = self.backtests[0].total_returns
        benchmark_returns = self.backtests[0].benchmark_returns
        joint_stock_codes = [self.backtests[0].test_env.cur_stock_code]
        ##calculate average metrics and returns
        for i in range(1, len(self.backtests)):
            mean_metrics += self.backtests[i].evaluate()
            mean_total_returns += self.backtests[i].total_returns
            joint_stock_codes.append(self.backtests[i].test_env.cur_stock_code)

        self.mean_metrics = mean_metrics / len(self.backtests)
        self.mean_total_returns = mean_total_returns ##don't need to average here because of the cash allocation
        self.joint_stock_codes = ','.join(joint_stock_codes)
        self.benchmark_returns = self.backtests[0].benchmark_returns


    def show_result(self):
        self.evaluate()
        dates = pd.date_range(start=test_start_date_str, end=test_end_date_str)  # Example dates

        # Create figure
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(x=dates, y=self.mean_total_returns,  mode='lines', name=f'Strategy ({self.joint_stock_codes})',
                                fill='tozeroy', fillcolor='rgba(0,100,100,0.2)'))
        fig.add_trace(go.Scatter(x=dates, y=self.benchmark_returns, mode='lines', name=f'Benchmark ({self.backtests[0].benchmark_stock})',
                                fill='tozeroy', fillcolor='rgba(200,0,80,0.2)'))
        # fig.add_trace(go.Scatter(x=dates, y=stock_returns, mode='lines', name=f'Stock Price ({self.test_env.cur_stock_code})',
        #                         fill='tozeroy', fillcolor='rgba(0,200,80,0.2)'))
        # fig.add_trace(go.Scatter(x=dates, y=self.portfolio_trace, name='Cash Pool Usage'), row=2, col=1)

        fig.add_annotation(x=0, y=1.25, xref="paper", yref="paper",
                          text=
                          f"<span style='font-size: 12px; color: grey;'>Total Return:</span> <b>{self.mean_total_returns[-1]*100:.2f}%</b>        " +
                          # f"<span style='font-size: 12px; color: grey;'>Annual Return:</span> <b>{2*100:.2f}%</b>        " +
                          f"<span style='font-size: 12px; color: grey;'>Benchmark Return:</span> <b>{self.benchmark_returns[-1]*100:.2f}%</b>        " +
                          f"<span style='font-size: 12px; color: grey;'>Alpha:</span> <b>{self.mean_metrics[-1]:.2f}</b>        " +
                          f"<span style='font-size: 12px; color: grey;'>Beta:</span> <b>{self.mean_metrics[-2]:.2f}</b>        " +
                          f"<span style='font-size: 12px; color: grey;'>Sharp Ratio:</span> <b>{self.mean_metrics[1]:.2f}</b>",
                          showarrow=False, font=dict(size=15), borderwidth=2, borderpad=4)

        fig.add_annotation(x=0, y=1.15, xref="paper", yref="paper",
                          text=
                          f"<span style='font-size: 12px; color: grey;'>Win Rate:</span> <b>{self.mean_metrics[2] * 100:.2f}%\n</b>         " +
                          f"<span style='font-size: 12px; color: grey;'>Max Drawdown:</span> <b>{self.mean_metrics[5]* 100:.2f}%</b>         " +
                          f"<span style='font-size: 12px; color: grey;'>Information Ratio:</span> <b>{self.mean_metrics[6]:.2f}</b>         " +
                          f"<span style='font-size: 12px; color: grey;'>Return Volatility:</span> <b>{self.mean_metrics[4]:.2f}</b>         " +
                          f"<span style='font-size: 12px; color: grey;'>Profit Loss Ratio:</span> <b>{self.mean_metrics[3]:.2f}</b>" ,
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
        )

        # Show figure
        fig.show()

