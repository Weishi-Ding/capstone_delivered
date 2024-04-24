#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Author: Zichen Yang
#Date: 09-17-2023
#Assignment: RL HW01

import numpy as np
import random
import matplotlib.pyplot as plt

std_dev   = 1
rw_dev    = 0.01
std_mean  = 0
step_size = 0.1
epsilon   = 0.1   ##prbability of exploration
decay_rate = 0.00008
num_runs = 2000
num_steps = 10000


# # Environment Class

# In[ ]:


class Environment:
    def __init__(self, k = 10):
        ###sample_average
        self.num_arms = k
        self.qs = np.zeros(k)          ##This is q*(a)

    ##Reset function is used to create a new k-armed Testbed
    def reset(self):
        self.qs = self.qs = np.zeros(self.num_arms)           ##This is q*(a)


    ##Call this function to take one action
    def bandit(self, action):
        reward = np.random.normal(self.qs[action], std_dev)
        for i in range(self.num_arms):
            self.qs[i] +=  np.random.normal(std_mean, rw_dev) ##update q*(a)
        return reward

    def get_optimal_actions(self):
        max_val = max(self.qs)
        indices = [i for i, x in enumerate(self.qs) if x == max_val]
        return indices


# # Agent Class

# In[ ]:


class Agent:
    def __init__(self, k = 10):
        self.num_arms = k
        self.Q_values = np.zeros(k)  ##This is Q(A)
        self.action_count = np.zeros(k)  ##This is N(A)

        ###Gradient-Bandit
        self.performances = np.zeros(k)  ##This is performance
        self.rewards = [0]           ##Initialize as [0] to avoid NaN


    def reset(self):
        self.Q_values = np.zeros(self.num_arms)  ##This is Q(A)
        self.action_count = np.zeros(self.num_arms)  ##This is N(A)

        ###Gradient-Bandit
        self.performances = np.zeros(self.num_arms)  ##This is performance
        self.rewards = [0]

    def pick_action_greedy(self):
        max_val = max(self.Q_values)
        indices = [i for i, x in enumerate(self.Q_values) if x == max_val]
        return random.choice(indices)

    ##Call this function to pick one action using sample-average
    def pick_action_eg(self, epsilon):
        num = random.random()
        if num <= epsilon:
            action = np.random.randint(0, self.num_arms)
            return action
        else:
            return self.pick_action_greedy()

    def pick_action_ucb(self, c, time_step):
        ucb_values = self.Q_values + c * np.sqrt(np.log(time_step + 1) / (self.action_count + 1e-5))
        max_val = max(ucb_values)
        indices = [i for i, x in enumerate(ucb_values) if x == max_val]
        return random.choice(indices)

    ##Call this function to pick one action using gradient-bandit
    def pick_action_gb(self):
        prob_a = np.exp(self.performances) / np.sum(np.exp(self.performances))
        action = np.random.choice([x for x in range(self.num_arms)], p=prob_a)
        return actionper

     ##Call this function to update the true values and action counts using sample average method
    def sample_average_update(self, action, reward):
        self.action_count[action] += 1
        self.Q_values[action] = self.Q_values[action] + (1 / self.action_count[action]) * (reward - self.Q_values[action])
        self.rewards.append(reward)

    def step_size_update(self, step_size, action, reward):
        self.action_count[action] += 1
        self.Q_values[action] = self.Q_values[action] + step_size * (reward - self.Q_values[action])
        self.rewards.append(reward)

    def increase_step_size_update(self, decay_rate, step_size, action, reward, time_step):
        new_step_size = step_size / (1 - decay_rate * time_step)
        self.step_size_update(new_step_size, action, reward)

    ##Call this function to update the true values and action counts using gradient method
    def gradient_update(self, action, reward):
        prob_a = np.exp(self.performances) / np.sum(np.exp(self.performances))

        for i in range(self.num_arms):
            if action == i:
                self.performances[i] += step_size * (reward - np.mean(self.rewards)) * (1 - prob_a[i])
            else:
                self.performances[i] -= step_size * (reward - np.mean(self.rewards)) * prob_a[i]
        self.rewards.append(reward)


# # Main Program

# In[ ]:


#### sample-average method with greedy as the bench mark
player = Agent()     ##Create an agent
game = Environment() ##Create a game
sum_ave_rewards = np.zeros(num_steps)
sum_opt_per     = np.zeros(num_steps)

for i in range(num_runs):
    opt_list = []
    for j in range(num_steps):
        opt_actions = game.get_optimal_actions()
        action = player.pick_action_greedy()
        if action in opt_actions:
            opt_list.append(1)
        else:
            opt_list.append(0)
        reward = game.bandit(action)
        player.sample_average_update(action, reward)

    ave_rewards = np.cumsum(player.rewards[1:]) / np.arange(1, num_steps + 1)
    opt_per = np.cumsum(opt_list) / np.arange(1, num_steps + 1) * 100

    sum_ave_rewards += ave_rewards
    sum_opt_per += opt_per

    player.reset()
    game.reset()

overall_ave_rewards = sum_ave_rewards / num_runs
overall_opt_per     = sum_opt_per / num_runs


# In[ ]:


#### constant step-size method with UCB
player2 = Agent()     ##Create an agent
game2 = Environment() ##Create a game
sum_ave_rewards2 = np.zeros(num_steps)
sum_opt_per2     = np.zeros(num_steps)

for i in range(num_runs):
    opt_list = []
    for j in range(num_steps):
        opt_actions = game2.get_optimal_actions()
        action = player2.pick_action_ucb(epsilon, j)
        if action in opt_actions:
            opt_list.append(1)
        else:
            opt_list.append(0)
        reward = game2.bandit(action)
        player2.step_size_update(step_size, action, reward)

    ave_rewards = np.cumsum(player2.rewards[1:]) / np.arange(1, num_steps + 1)
    opt_per = np.cumsum(opt_list) / np.arange(1, num_steps + 1) * 100

    sum_ave_rewards2 += ave_rewards
    sum_opt_per2 += opt_per

    player2.reset()
    game2.reset()

overall_ave_rewards2 = sum_ave_rewards2 / num_runs
overall_opt_per2     = sum_opt_per2 / num_runs


# In[ ]:


#### sample-average method with epsilon greedy
player3 = Agent()     ##Create an agent
game3 = Environment() ##Create a game
sum_ave_rewards3 = np.zeros(num_steps)
sum_opt_per3     = np.zeros(num_steps)

for i in range(num_runs):
    opt_list = []
    for j in range(num_steps):
        opt_actions = game3.get_optimal_actions()
        action = player3.pick_action_eg(epsilon)
        if action in opt_actions:
            opt_list.append(1)
        else:
            opt_list.append(0)
        reward = game3.bandit(action)
        player3.sample_average_update(action, reward)

    ave_rewards = np.cumsum(player3.rewards[1:]) / np.arange(1, num_steps + 1)
    opt_per = np.cumsum(opt_list) / np.arange(1, num_steps + 1) * 100

    sum_ave_rewards3 += ave_rewards
    sum_opt_per3 += opt_per

    player3.reset()
    game3.reset()

overall_ave_rewards3 = sum_ave_rewards3 / num_runs
overall_opt_per3     = sum_opt_per3 / num_runs


# In[ ]:


##increasing step-size with UCB
player4 = Agent()     ##Create an agent
game4 = Environment() ##Create a game
sum_ave_rewards4 = np.zeros(num_steps)
sum_opt_per4     = np.zeros(num_steps)

for i in range(num_runs):
    opt_list = []
    for j in range(num_steps):
        opt_actions = game4.get_optimal_actions()
        action = player4.pick_action_ucb(epsilon,j)
        if action in opt_actions:
            opt_list.append(1)
        else:
            opt_list.append(0)
        reward = game4.bandit(action)
        player4.increase_step_size_update(decay_rate, step_size, action, reward, j)

    ave_rewards = np.cumsum(player4.rewards[1:]) / np.arange(1, num_steps + 1)
    opt_per = np.cumsum(opt_list) / np.arange(1, num_steps + 1) * 100

    sum_ave_rewards4 += ave_rewards
    sum_opt_per4 += opt_per

    player4.reset()
    game4.reset()

overall_ave_rewards4 = sum_ave_rewards4 / num_runs
overall_opt_per4     = sum_opt_per4 / num_runs


# In[ ]:


##gradient bandit
player5 = Agent()     ##Create an agent
game5 = Environment() ##Create a game
sum_ave_rewards5 = np.zeros(num_steps)
sum_opt_per5     = np.zeros(num_steps)

for i in range(num_runs):
    opt_list = []
    for j in range(num_steps):
        opt_actions = game5.get_optimal_actions()
        action = player5.pick_action_gb()
        if action in opt_actions:
            opt_list.append(1)
        else:
            opt_list.append(0)
        reward = game5.bandit(action)
        player5.gradient_update(action, reward)

    ave_rewards = np.cumsum(player5.rewards[1:]) / np.arange(1, num_steps + 1)
    opt_per = np.cumsum(opt_list) / np.arange(1, num_steps + 1) * 100

    sum_ave_rewards5 += ave_rewards
    sum_opt_per5 += opt_per

    player5.reset()
    game5.reset()

overall_ave_rewards5 = sum_ave_rewards5 / num_runs
overall_opt_per5     = sum_opt_per5 / num_runs


# # Graphs for the Main Experiment

# In[ ]:


###Plot graphs
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_steps), overall_ave_rewards, label = "sample-average with ε-greedy(ε=0)")
plt.plot(np.arange(num_iterations), overall_ave_rewards2, label = "constant step-size with UCB")
plt.plot(np.arange(num_iterations), overall_ave_rewards3, label = "sample-average with ε-greedy(ε=0.1)")
plt.xticks([0, 2000, 4000, 6000, 8000, 10000], ["0", "2000", "4000", "6000", "8000", "10000"], fontsize = 12)
plt.xlabel("Steps", fontsize = 15)
plt.ylabel("Average\nreward", fontsize = 15, rotation="horizontal", labelpad=40)
plt.title("Average Reward Over 2000 Runs", fontsize = 20)
plt.legend(fontsize = 16)


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_iterations), overall_opt_per, label = "sample-average with ε-greedy(ε=0)")
plt.plot(np.arange(num_iterations), overall_opt_per2, label = "constant step-size with UCB")
plt.plot(np.arange(num_iterations), overall_opt_per3, label = "sample-average with ε-greedy(ε=0.1)")
plt.xticks([0, 2000, 4000, 6000, 8000, 10000], ["0", "2000", "4000", "6000", "8000", "10000"], fontsize = 12)
plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], fontsize = 12)
plt.xlabel("Steps", fontsize = 15)
plt.ylabel("%Optimal\naction", fontsize = 15, rotation="horizontal", labelpad=40)
plt.title("Average %Optimal_Action Over 2000 Runs", fontsize = 20)
plt.legend(fontsize = 16)


# # Graphs for Additional Questions

# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_steps), overall_ave_rewards, label = "sample-average with ε-greedy(ε=0)")
plt.plot(np.arange(num_iterations), overall_ave_rewards2, label = "constant step-size with UCB")
plt.plot(np.arange(num_iterations), overall_ave_rewards3, label = "sample-average with ε-greedy(ε=0.1)")
plt.plot(np.arange(num_iterations), overall_ave_rewards4, label = "constant step-size with UCB")
plt.plot(np.arange(num_iterations), overall_ave_rewards5, label = "Gradient-Bandit Algorithm")
plt.xticks([0, 2000, 4000, 6000, 8000, 10000], ["0", "2000", "4000", "6000", "8000", "10000"], fontsize = 12)
plt.xlabel("Steps", fontsize = 15)
plt.ylabel("Average\nreward", fontsize = 15, rotation="horizontal", labelpad=40)
plt.title("Average Reward Over 2000 Runs", fontsize = 20)
plt.legend(fontsize = 16)


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_iterations), overall_opt_per, label = "sample-average with ε-greedy(ε=0)")
plt.plot(np.arange(num_iterations), overall_opt_per2, label = "constant step-size with UCB")
plt.plot(np.arange(num_iterations), overall_opt_per3, label = "sample-average with ε-greedy(ε=0.1)")
plt.plot(np.arange(num_iterations), overall_opt_per4, label = "increasing step-size with UCB")
plt.plot(np.arange(num_iterations), overall_opt_per5, label = "Gradient-Bandit Algorithm")
plt.xticks([0, 2000, 4000, 6000, 8000, 10000], ["0", "2000", "4000", "6000", "8000", "10000"], fontsize = 12)
plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], fontsize = 12)
plt.xlabel("Steps", fontsize = 15)
plt.ylabel("%Optimal\naction", fontsize = 15, rotation="horizontal", labelpad=40)
plt.title("Average %Optimal_Action Over 2000 Runs", fontsize = 20)
plt.legend(fontsize = 16)


# In[ ]:




