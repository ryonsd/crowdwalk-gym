# -*- coding: utf-8 -*-
import numpy as np
import argparse
import gym
import sys
sys.path.append("../../")
import CrowdWalkGym

env = gym.make("two-routes-v0")

episodes = 1
max_steps = 10000


# train
for e_i in range(1, episodes+1):
    print("="*40)
    print("episode:", e_i)
    e_step = 0
    e_reward = 0

    env.reset()
    while e_step < max_steps:
        # step

        e_step += 1