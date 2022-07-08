# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import pandas as pd
import os
import json
import sys
from two_routes import TwoRoutesEnv

# リンク上の各時刻の歩行者数および密度
def get_obs_reward(env, log, step_duration, step, sample_t = 10):
    link_dict = {} # length, width, density of each link
    current_state_dict = {} # number of pedestrians at each link

    T = int(step_duration / sample_t)
    link_dict = {}
    current_state_dict = {}
    for link_name, link_attribute in env.link.items():
        if link_name != "generation_link":
            link_dict[link_name] = {"link_list": link_attribute["id"], "length": link_attribute["width"], "width": link_attribute["width"], 
                                    "n_ped": np.zeros(T), "density": np.zeros(T)}
        current_state_dict[link_name] = 0

    check_step = np.linspace(step - step_duration + sample_t, step, T)

    for t, s in enumerate(check_step):
        log_sample = log[log.current_traveling_period == s].reset_index(drop=True)
        
        for i in range(len(log_sample)):
            if not np.isnan(log_sample.current_traveling_period[i]):
                l = log_sample.current_linkID[i]
                
                for link_name, link_attribute in link_dict.items():
                    if l in link_attribute["link_list"]:
                        link_attribute["n_ped"][t] += 1
                        link_attribute["density"][t] += 1 / (link_attribute["length"]* link_attribute["width"])
                        if t == T-1:
                            current_state_dict[link_name] += 1

    state = list(current_state_dict.values())

    # 混雑度
    congestion_degree = 0
    for link_name, link_attribute in link_dict.items():
        congestion_degree += sum(link_attribute["density"] > 1.08).astype(int)

    return state, int(congestion_degree)
    


if __name__ == '__main__':
    args = sys.argv

    env_name = args[1]
    step_duration = int(args[2])
    step = int(args[3])
    log_dir = args[4] + "/log/"

    if env_name == "two_routes":
        env = TwoRoutesEnv()

    log = pd.read_csv(log_dir + "log_individual_pedestrians.csv")

    state, reward = get_obs_reward(env, log, step_duration, step)

    history = {"step": step, "state": state, "reward": reward}
    with open(log_dir + "history.json", "w") as f:
        json.dump(history, f, indent=2)


    