# -*- coding: utf-8 -*-
import json
import sys
import numpy as np

if __name__ == '__main__':

    args = sys.argv

    step = int(args[1])
    logdir = args[2]

    # print("check")
    # get state
    is_step = False
    while not is_step:
        try:
            with open(logdir + "/history.json", "r") as f:
                history = json.load(f)
            action = history[str(step)]["action"]
            # print(action)
            if not np.isnan(action):
                is_step = True
        except:
            pass

    # print("get action", action)
