# -*- coding: utf-8 -*-
import numpy as np
import argparse
import gym
import sys
sys.path.append("../../")
import CrowdWalkGym

env = gym.make("two-routes-v0")

episodes = 2
max_steps = 10000
prop_file = "/home/nishida/CrowdWalk_nsd/crowdwalk/sample/2routes/properties.json"

# train
for e_i in range(1, episodes+1):
    print("="*40)
    print("episode:", e_i)
    e_step = 0
    e_reward = 0

    env.reset(prop_file)
    while e_step < max_steps:
        # step

        e_step += 1