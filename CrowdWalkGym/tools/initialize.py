# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import pandas as pd
import os
import json
import sys

if __name__ == '__main__':
    

    args = sys.argv

    path_to_gym = args[1]
    env_name = args[2]
    agent_log_dir = args[3]
    sim_dir = args[4]

    import sys
    sys.path.append((path_to_gym+"envs/"))
    from two_routes import TwoRoutesEnv
    from moji import MojiEnv

    if env_name == "two_routes":
        env = TwoRoutesEnv()
    elif env_name == "moji":
        env = MojiEnv()

    gen = pd.read_csv(sim_dir + "/generation.csv")
    generation_pedestrian_number = gen[gen.step == 0]["n_ped"].values[0]

    init_ped_num = int(generation_pedestrian_number)
    state = list(np.append(np.array([init_ped_num]), np.zeros(env.nS)))

    history = {}
    step = 0
    history[step] = {"sim_step": 0, "state": state, "action": np.nan, "reward": np.nan, "next_state": np.nan, "done": np.nan}
    with open(agent_log_dir + "history.json", "w") as f:
        json.dump(history, f, indent=2)

    agent_dict = {}
    with open(agent_log_dir + "agent_dict.json", "w") as f:
        json.dump(agent_dict, f,  indent=2, ensure_ascii=False)

    