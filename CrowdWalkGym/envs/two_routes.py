# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import pandas as pd
import gym
import os
import json
import sys

import warnings
warnings.simplefilter('ignore')

class TwoRoutesEnv(gym.Env):
    def __init__(self, is_gui=False):
        self.is_gui = is_gui

        self.nS = 13
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(14,))
        
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

        self.route1_length = 0.4
        self.route2_length = 0.7

    def prepare(self, path_to_crowdwalk_dir, path_to_gym, path_to_run_dir, n_obj):
        sys.path.append(path_to_gym)
        from tools import create_properties_file

        self.path_to_crowdwalk_dir = path_to_crowdwalk_dir
        path_to_crowdwalk_config_dir = path_to_crowdwalk_dir + "sample/two_routes/"
        self.path_to_run_dir = path_to_run_dir
        self.prop_file = path_to_run_dir + "/properties.json"

        create_properties_file.do(path_to_crowdwalk_config_dir, path_to_gym, path_to_run_dir, n_obj)

    def reset(self):
        if os.path.isfile(self.path_to_run_dir + "/history.json"):
            os.remove(self.path_to_run_dir + "/history.json")

        if self.is_gui:
            subprocess.Popen(["sh", self.path_to_crowdwalk_dir+"quickstart.sh", self.prop_file, "-lError"], stderr=subprocess.DEVNULL)
            # subprocess.Popen(["sh", self.path_to_crowdwalk_dir+"quickstart.sh", self.prop_file, "-lError"])
        else:
            # subprocess.Popen(["sh", self.path_to_crowdwalk_dir+"quickstart.sh", self.prop_file, "-c", "-lError"], stderr=subprocess.DEVNULL)
            subprocess.Popen(["sh", self.path_to_crowdwalk_dir+"quickstart.sh", self.prop_file, "-c", "-lError"])
        return np.zeros(self.nS)

    def step(self):


        return next_state, reward, done, {}
