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
from DQN import DQN, Experience_Buffer

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

class Agent():
    def __init__(self, model=sNone, ):
        self.state = None
        self.portfolio = 0
        self.input_dim = len(train_data.columns)-2 ##-2 for trade_date and pctc_change columns
        self.output_dim = output_dim 
        self.action_space = action_space
        if not model:
          self.policy_net = DQN(self.input_dim, self.output_dim)
          self.target_net = DQN(self.input_dim, self.output_dim)
        else:
          self.policy_net = model
          self.target_net = model
        ##copy the policy net and set target net to eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.exp_buffer = Experience_Buffer(buffer_cap=1500)

    def get_action(self, state, epsilon):
        action = 0
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.output_dim)
        else:
            res = self.policy_net(state)
            val, indices = torch.max(res, 0)
            action = indices.item()
        return action

    def validation(self, val_env,device):
        self.policy_net.to(device)
        self.target_net.to(device)
        exp_buffer = Experience_Buffer(buffer_cap=1500)
        loss_fn = nn.MSELoss()
        state = val_env.reset() ##get the start state
        state = np.append(state, self.portfolio)
        terminated = False
        time_step = 0
        ##set the policy net to train mode
        self.policy_net.eval()
        self.target_net.eval()

        val_loss = 0
        with torch.no_grad():
          while not terminated:
              action_idx = self.get_action(torch.tensor(state).to(device).float(), 0)
              action = self.action_space[action_idx]
              next_state, reward, terminated, new_portfolio = val_env.step(action, self.portfolio)
              if terminated:
                  ##ensure that the experience for termination is not added to the buffer
                  break
              next_state = np.append(next_state, new_portfolio)
              self.portfolio = new_portfolio
              self.exp_buffer.add(Experience(state, action_idx, reward, next_state))
              state = next_state

              if self.exp_buffer.can_provide_sample(batch_size) and time_step % 3 == 0:
                  experiences = self.exp_buffer.sample(batch_size)
                  states, actions, rewards, next_states = extract_tensors(experiences, batch_size, self.input_dim)
                  states = states.to(device)
                  actions = actions.to(device)
                  rewards = rewards.to(device)
                  next_states = next_states.to(device)

                  current_q_values = get_cur_Qs(self.policy_net, states, actions)
                  target_q_values = get_target_Qs(self.target_net, next_states)
                  target_q_values = (target_q_values * gamma) + rewards

                  loss = loss_fn(current_q_values, target_q_values)
                  val_loss += loss.item()
              time_step += 1

        return val_loss / max(time_step // 3, 1)


    def train(self, train_env, val_env, num_episode, batch_size, epsilon, gamma, lr, device):
        self.policy_net.to(device)
        self.target_net.to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        train_loss = []
        val_loss = []

        for episode in range(num_episode):
            state = train_env.reset() ##get the start state
            state = np.append(state, self.portfolio)
            terminated = False
            time_step = 0
            ##set the policy net to train mode
            self.policy_net.train()

            t_loss = 0
            while not terminated:
                action_idx = self.get_action(torch.tensor(state).to(device).float(), epsilon)
                action = self.action_space[action_idx]
                next_state, reward, terminated, new_portfolio = train_env.step(action, self.portfolio)
                if terminated:
                    ##ensure that the experience for termination is not added to the buffer
                    break
                next_state = np.append(next_state, new_portfolio)
                self.portfolio = new_portfolio
                self.exp_buffer.add(Experience(state, action_idx, reward, next_state))
                state = next_state

                if self.exp_buffer.can_provide_sample(batch_size) and time_step % 3 == 0:
                    experiences = self.exp_buffer.sample(batch_size)
                    states, actions, rewards, next_states = extract_tensors(experiences, batch_size, self.input_dim)
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    next_states = next_states.to(device)

                    current_q_values = get_cur_Qs(self.policy_net, states, actions)
                    target_q_values = get_target_Qs(self.target_net, next_states)
                    target_q_values = (target_q_values * gamma) + rewards

                    loss = loss_fn(current_q_values, target_q_values)
                    t_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                time_step += 1

            if episode % target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            #validation
            val_loss.append(self.validation(val_env, device))
            train_loss.append(t_loss / max(time_step // 3, 1))
            print(f"Episode: {episode}, Train Loss: {train_loss[-1]}, Val Loss: {val_loss[-1]}")

        return train_loss, val_loss





