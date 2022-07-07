# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import pandas as pd
import gym
import os
import json

class TwoRoutesEnv(gym.Env):
    def __init__(self):
        self.path_to_crowdwalk= "/home/nishida/CrowdWalk_nsd/crowdwalk/"
        self.is_gui = True

        self.nS = 13
        self.nA = 2

    def reset(self, prop_file):
        if self.is_gui:
            subprocess.run(["sh", self.path_to_crowdwalk+"quickstart.sh", prop_file, "-lError"])
        else:
            subprocess.run(["sh", self.path_to_crowdwalk+"quickstart.sh", prop_file, "-c", "-lError"])
        return np.zeros(self.nS)

    def step(self):


        return next_state, reward, done, {}