"""
DQN pytorch implementation is based on
Andrew Gordienko: Reinforcement Learning: DQN w Pytorch
https://andrew-gordienko.medium.com/reinforcement-learning-dqn-w-pytorch-7c6faad3d1e
"""


import numpy as np
import os
import json
import sys
import random
import datetime
from itertools import count

import gym
sys.path.append("../")
import CrowdWalkGym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import tensorboardX as tb


class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones


class Network(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, output_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class DQN_Solver:
    def __init__(self, nS, nA):
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.network = Network(input_shape=nS, output_shape=nA)

    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return env.action_space.sample()
        
        state = torch.tensor(observation).float().detach()
        state = state.to(DEVICE)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        return torch.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return 0
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        
        q_target = rewards + GAMMA * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        return loss.item()

    def returning_epsilon(self):
        return self.exploration_rate


def step(e_step):
    # get state
    is_step = False
    while not is_step:
        try:
            with open("log/history.json", "r") as f:
                history = json.load(f)
            state = history[str(e_step)]["state"]
            is_step = True
        except:
            continue
    
    # take action
    action = agent.choose_action(state)

    history[str(e_step)]["action"] = action
    with open("log/history.json", "w") as f:
        json.dump(history, f)

    # step
    # next_state, reward, done
    is_step = False
    while not is_step:
        try:
            with open("log/history.json", "r") as f:
                history = json.load(f)
            if not np.isnan(history[str(e_step)]["next_state"]).sum():
                is_step = True
        except:
            continue
    
    next_state = history[str(e_step)]["next_state"]
    reward = history[str(e_step)]["reward"] / 150
    done = history[str(e_step)]["done"]

    return state, action, next_state, reward, done
    

##########################################################################
if __name__ == '__main__':
    env = gym.make("two-routes-v0")

    logdir = "log/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = tb.SummaryWriter(logdir=logdir)

    FC1_DIMS = 1024
    FC2_DIMS = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPISODES = 1000
    LEARNING_RATE = 0.00001 
    MEM_SIZE = 10000
    BATCH_SIZE = 32
    GAMMA = 1.0
    EXPLORATION_MAX = 1.0
    EXPLORATION_DECAY = 0.995
    EXPLORATION_MIN = 0.001
    LOG_EPISODE = 1

    agent = DQN_Solver(env.nS+1, env.nA)

    # train
    num_episodes = 10000
    total_steps = 0

    if os.path.isfile("log/history.json"):
        os.remove("log/history.json")

    for e_i in range(1, EPISODES+1):
        e_step = 0
        e_reward = 0
        e_loss = 0
        done = False

        if os.path.isfile("log/history.json"):
            os.remove("log/history.json")

        env.reset()

        while True:

            state, action, next_state, reward, done = step(e_step)
            agent.memory.add(state, action, reward, next_state, done)
            e_loss += agent.learn()

            e_step += 1
            e_reward += reward
            total_steps += 1

            if done:
                print("episode:", e_i, "episode_step:", e_step, "total_steps:", total_steps, "reward:", "{:.2f}".format(e_reward), "loss:", "{:.3f}".format(e_loss/e_step))
                break

        if e_i % LOG_EPISODE == 0:
            writer.add_scalar('epsilon', agent.returning_epsilon(), e_i)
            writer.add_scalar('loss', e_loss / e_step, e_i)
            writer.add_scalar('total_reward', e_reward, e_i)

        
            