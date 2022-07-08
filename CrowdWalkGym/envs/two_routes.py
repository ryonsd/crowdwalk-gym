# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import pandas as pd
import gym
import os
import json

import warnings
warnings.simplefilter('ignore')

class TwoRoutesEnv(gym.Env):
    def __init__(self):
        self.path_to_crowdwalk= "/home/nishida/CrowdWalk_nsd/crowdwalk/" # <<<<<<<<<<<<<<<<<<
        self.prop_file = self.path_to_crowdwalk + "sample/two_routes/properties.json"
        self.is_gui = False

        self.nS = 13
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(13,))
        
        self.nA = 2
        self.action_space = gym.spaces.Discrete(2)

        self.link = {
            "start_link": {"id": "_p00011" ,"length": 100,  "width": 2},

            "route1_1": {"id": "_p00004" ,"length": 100,  "width": 2},
            "route1_2": {"id": "_p00006" ,"length": 100,  "width": 2},
            "route1_3": {"id": "_p00024" ,"length": 100,  "width": 2},
            "route1_4": {"id": "_p00025" ,"length": 100,  "width": 1},

            "route2_1": {"id": "_p00021" ,"length": 100,  "width": 2},
            "route2_2": {"id": "_p00028" ,"length": 100,  "width": 2},
            "route2_3": {"id": "_p00027" ,"length": 100,  "width": 2},
            "route2_4": {"id": "_p00016" ,"length": 100,  "width": 2},
            "route2_5": {"id": "_p00018" ,"length": 100,  "width": 2},
            "route2_6": {"id": "_p00020" ,"length": 100,  "width": 2},
            "route2_7": {"id": "_p00022" ,"length": 100,  "width": 2},

            "goal_link": {"id": "_p00010" ,"length": 200,  "width": 2},
        }

    def reset(self):
        if self.is_gui:
            subprocess.run(["sh", self.path_to_crowdwalk+"quickstart.sh", self.prop_file, "-lError"])
        else:
            subprocess.run(["sh", self.path_to_crowdwalk+"quickstart.sh", self.prop_file, "-c", "-lError"])
        return np.zeros(self.nS)

    def step(self):


        return next_state, reward, done, {}
