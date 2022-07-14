# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import pandas as pd
import os
import json
import sys

def get_reward(env, log, step_s, step_e, sample_t = 10, n_obj=1):
    link_dict = {} # length, width, density of each link

    step_duration = step_e - step_s
    T = int(step_duration / sample_t)
    link_dict = {}
    for link_name, link_attribute in env.link.items():
        if link_name != "generation_link":
            link_dict[link_name] = {"link_list": link_attribute["id"], "length": link_attribute["length"], "width": link_attribute["width"], 
                                    "n_ped": np.zeros(T), "density": np.zeros(T)}

    check_step = np.arange(step_s+sample_t, step_e, sample_t)

    for t, s in enumerate(check_step):
        log_sample = log[log.current_traveling_period == s].reset_index(drop=True)
        
        for i in range(len(log_sample)):
            if not np.isnan(log_sample.current_traveling_period[i]):
                l = log_sample.current_linkID[i]
                
                for link_name, link_attribute in link_dict.items():
                    if l in link_attribute["link_list"]:
                        link_attribute["n_ped"][t] += 1
                        link_attribute["density"][t] += 1 / (link_attribute["length"]* link_attribute["width"])
    
    # 混雑度
    congestion_degree = 0
    for link_name, link_attribute in link_dict.items():
        congestion_degree += sum(link_attribute["density"] > 0.71) #.astype(int)

    if n_obj == 1:
        reward = -int(congestion_degree)
    # else:
    #     reward = []

    done = True

    return reward, done



if __name__ == '__main__':

    args = sys.argv

    path_to_gym = args[1]
    env_name = args[2]
    step = int(args[3])
    sim_previous_step = int(args[4])
    sim_final_step = int(args[5])
    sim_log_dir = args[6] + "/log/"
    agent_log_dir = args[7]

    import sys
    sys.path.append((path_to_gym+"envs/"))
    from two_routes import TwoRoutesEnv

    if env_name == "two_routes":
        env = TwoRoutesEnv()

    log = pd.read_csv(sim_log_dir + "log_individual_pedestrians.csv")

    reward, done = get_reward(env, log, sim_previous_step, sim_final_step)

    with open(agent_log_dir + "history.json", "r") as f:
        history = json.load(f)

    history[str(step)]["reward"] = reward
    history[str(step)]["next_state"] = list(np.zeros(env.nS+1))
    history[str(step)]["done"] = done

    with open(agent_log_dir + "history.json", "w") as f:
        json.dump(history, f, indent=2)