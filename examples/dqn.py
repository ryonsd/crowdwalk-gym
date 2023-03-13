"""
DQN pytorch implementation is based on
Andrew Gordienko: Reinforcement Learning: DQN w Pytorch
site: https://andrew-gordienko.medium.com/reinforcement-learning-dqn-w-pytorch-7c6faad3d1e
GitHub: https://github.com/AndrewGordienko/Reinforcement-Learning/blob/master/dqn.py
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
import crowdwalk_gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from timm.scheduler import CosineLRScheduler

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

        self.optimizer = optim.Adam(self.parameters())#, lr=LEARNING_RATE)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=4000, lr_min=1e-7, 
                                  warmup_t=500, warmup_lr_init=5e-5, warmup_prefix=True)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class DQN_agent:
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
    
    def learn(self, t):
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
        self.network.scheduler.step(t+1)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        return loss.item()

    def returning_epsilon(self):
        return self.exploration_rate


def step(e_step, path_to_run_dir):
    # get state
    is_step = False
    while not is_step:
        try:
            with open(path_to_run_dir + "/history.json", "r") as f:
                history = json.load(f)
            state = history[str(e_step)]["state"]
            is_step = True
        except:
            continue
    
    # take action
    action = agent.choose_action(state)

    history[str(e_step)]["action"] = action
    with open(path_to_run_dir + "/history.json", "w") as f:
        json.dump(history, f)

    # step
    # next_state, reward, done
    is_step = False
    while not is_step:
        try:
            with open(path_to_run_dir + "/history.json", "r") as f:
                history = json.load(f)
            if not np.isnan(history[str(e_step)]["next_state"]).sum():
                is_step = True
        except:
            continue
    
    next_state = history[str(e_step)]["next_state"]
    reward = history[str(e_step)]["reward"] 
    done = history[str(e_step)]["done"]

    return state, action, next_state, reward, done
    

##########################################################################
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--env_name', default='two_routes')
    parser.add_argument('--gui', action='store_true')

    args = parser.parse_args()

    if args.env_name == "two_routes":
        env_id = "two-routes-v0"
    elif args.env_name == "moji":
        env_id = "moji-v0"
    elif args.env_name == "moji_small":
        env_id = "moji-v1"

    path_to_crowdwalk_dir = "/home/nishida/CrowdWalk_nsd/crowdwalk/"
    path_to_gym = os.path.abspath('..') + "/crowdwalk_gym/"
    path_to_run_dir = os.getcwd() + "/run/" + args.env_name + "/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(path_to_run_dir)

    env = gym.make(env_id, is_gui=args.gui)
    env.prepare(path_to_crowdwalk_dir, path_to_gym, path_to_run_dir, n_obj=1)

    writer = tb.SummaryWriter(logdir=path_to_run_dir)

    # parameters
    FC1_DIMS = 1024
    FC2_DIMS = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPISODES = 10000
    # LEARNING_RATE = 0.000005 #5e-6
    MEM_SIZE = 10000
    BATCH_SIZE = 32
    GAMMA = 1.0
    EXPLORATION_MAX = 1.0
    EXPLORATION_DECAY = 0.999
    EXPLORATION_MIN = 0.01
    LOG_EPISODE = 1
    OBJ_SIZE = 1

    agent = DQN_agent(env.nS+1, env.nA)

    # train
    total_steps = 0
    for e_i in range(1, EPISODES+1):
        e_step = 0
        e_reward = np.zeros(OBJ_SIZE)
        e_loss = 0
        done = False

        env.reset()
        while True:

            state, action, next_state, reward, done = step(e_step, path_to_run_dir)
            print("step", e_step, "state", state, "action", action, "next_state", next_state, "reward", reward, "done", done)
            agent.memory.add(state, action, reward/100, next_state, done)
            e_loss += agent.learn(total_steps)

            e_step += 1
            e_reward += np.array(reward)
            total_steps += 1

            if done:
                # print("episode:", e_i, "episode_step:", e_step, "total_steps:", total_steps, "reward:", "{:.2f}".format(e_reward), "loss:", "{:.3f}".format(e_loss/e_step))
                print("episode:", e_i, "episode_step:", e_step, "total_steps:", total_steps, "reward:", e_reward, "loss:", "{:.3f}".format(e_loss/e_step))
                break

        if e_i % LOG_EPISODE == 0:
            writer.add_scalar('lr', agent.network.scheduler.get_epoch_values(total_steps), e_i)
            writer.add_scalar('epsilon', agent.returning_epsilon(), e_i)
            writer.add_scalar('loss', e_loss / e_step, e_i)
            for i in range(OBJ_SIZE):
                writer.add_scalar('reward_'+str(i), e_reward[i], e_i)

            

        
            
