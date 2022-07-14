# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import json
import gym
import sys
import random
import math
import datetime
from collections import namedtuple, deque
from itertools import count

sys.path.append("../")
import CrowdWalkGym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import tensorboardX as tb


Transition = namedtuple('Transition',
                        ['state',
                         'action',
                         'reward',
                         'next_state',
                         'terminal'])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = self.fc4(x)
        return y



def optimize_model(gamma):
    if len(memory) < BATCH_SIZE:
        return torch.tensor(0)
    transitions = memory.sample(BATCH_SIZE)
 
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    non_final_mask = torch.tensor(tuple(map(lambda t: t is not True,
                                          batch.terminal)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for t, s in zip(batch.terminal, batch.next_state)
                                                if t is not True])
    # Predict Q values
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute the expected Q values
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = reward_batch + (gamma * next_state_values)

    # Compute Huber loss
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss



if __name__ == '__main__':
    env = gym.make("two-routes-v0")
    # logdir = "log/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logdir = "log/train_log"
    if os.path.isfile("log/history.json"):
        os.remove("log/history.json")
    writer = tb.SummaryWriter(logdir=logdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 300
    TARGET_UPDATE = 5
    LOG_EPISODE = 1

    n_state = env.nS + 1
    n_actions = env.nA

    policy_net = DQN(n_state, n_actions).to(device)
    target_net = DQN(n_state, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)
    memory = ReplayMemory(10000)

    # train
    num_episodes = 10000
    total_steps = 0
    for e_i in range(1, num_episodes+1):
        # print("="*40)
        # print("episode:", e_i)
        e_step = 0
        e_reward = 0
        e_loss = 0
        done = False

        if os.path.isfile("log/history.json"):
            os.remove("log/history.json")

        env.reset()

        while not done:
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
            state = list(map(float, state))
            state = torch.tensor(state).unsqueeze(dim=0) 

            # select action
            # epsilon-greedy
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * total_steps / EPS_DECAY)
            if random.random() > eps_threshold:
            # if 1.0 > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).argmax().unsqueeze(dim=0).unsqueeze(dim=0) 
            else:
                action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
            # print(action)

            # action = torch.tensor(1, device=device).unsqueeze(dim=0).unsqueeze(dim=0) 

            history[str(e_step)]["action"] = action.item()
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
            next_state = list(map(float, next_state))
            next_state = torch.tensor(next_state).unsqueeze(dim=0) 

            reward = history[str(e_step)]["reward"] / 150
            reward = torch.tensor([reward], device=device).unsqueeze(dim=0) 

            done = history[str(e_step)]["done"]

            # Store the transition in memory
            memory.push(state, action, reward, next_state, done)
            t = Transition(state=state, action=action, reward=reward, next_state=next_state, terminal=done)
            # print(e_step, t)

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model(GAMMA)
            # print(loss)
            e_loss += loss.item()

            e_step += 1
            e_reward += reward
            total_steps += 1

            if done:
                break

        if e_i % LOG_EPISODE == 0:
            writer.add_scalar('epsilon', eps_threshold, e_i)
            writer.add_scalar('loss', e_loss / e_step, e_i)
            writer.add_scalar('total_reward', e_reward, e_i)

        
        if e_i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print("episode:", e_i, "episode_step:", e_step, "total_steps:", total_steps, "reward:", "{:.2f}".format(e_reward.item()), "loss:", "{:.3f}".format(e_loss/e_step))
            