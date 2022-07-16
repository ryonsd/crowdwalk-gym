# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import pandas as pd
import gym
import os
import json

import warnings
warnings.simplefilter('ignore')

class MojiEnv(gym.Env):
    def __init__(self):
        self.path_to_crowdwalk= "/home/nishida/CrowdWalk_nsd/crowdwalk/" # <<<<<<<<<<<<<<<<<<
        self.prop_file = self.path_to_crowdwalk + "sample/moji/properties.json"
        self.is_gui = False

        self.nS = 19
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(19,))
        
        self.nA = 2
        self.action_space = gym.spaces.Discrete(2)

        self.link = {
            "start_link": {"id": "_p00001" ,"length": 50,  "width": 10},

            "route1_1": {"id": "_p00026" ,"length": 50,  "width": 10},
            "route1_2": {"id": "_p00027" ,"length": 50,  "width": 10},
            "route1_3": {"id": "_p00028" ,"length": 50,  "width": 10},
            "route1_4": {"id": "_p00029" ,"length": 50,  "width": 10},
            "route1_5": {"id": "_p00030" ,"length": 50,  "width": 10},
            "route1_6": {"id": "_p00076" ,"length": 50,  "width": 10},

            "route2_1": {"id": "_p00036" ,"length": 50,  "width": 10},
            "route2_2": {"id": "_p00037" ,"length": 50,  "width": 10},
            "route2_3": {"id": "_p00043" ,"length": 50,  "width": 10},
            "route2_4": {"id": "_p00045" ,"length": 50,  "width": 10},
            "route2_5": {"id": "_p00046" ,"length": 50,  "width": 10},
            "route2_6": {"id": "_p00047" ,"length": 50,  "width": 10},
            "route2_7": {"id": "_p00048" ,"length": 50,  "width": 10},
            "route2_8": {"id": "_p00064" ,"length": 50,  "width": 10},
            "route2_9": {"id": "_p00066" ,"length": 50,  "width": 10},
            "route2_10": {"id": "_p00070" ,"length": 50,  "width": 10},
            "route2_11": {"id": "_p00077" ,"length": 50,  "width": 10},

            "goal_link": {"id": "_p00003" ,"length": 100,  "width": np.inf},
        }

        self.route1_length = 0.3
        self.route2_length = 0.55

    def reset(self):
        if self.is_gui:
            # subprocess.Popen(["sh", self.path_to_crowdwalk+"quickstart.sh", self.prop_file, "-lError"], stderr=subprocess.DEVNULL)
            subprocess.Popen(["sh", self.path_to_crowdwalk+"quickstart.sh", self.prop_file, "-lError"])
        else:
            subprocess.Popen(["sh", self.path_to_crowdwalk+"quickstart.sh", self.prop_file, "-c", "-lError"], stderr=subprocess.DEVNULL)
        return np.zeros(self.nS)

    def step(self):


        return next_state, reward, done, {}
