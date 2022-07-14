# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import pandas as pd
import os
import json
import sys

# リンク上の各時刻の歩行者数および密度
def get_state_reward(env, log, step_duration, step, sample_t = 10, n_obj=1):
    link_dict = {} # length, width, density of each link
    next_state_dict = {} # number of pedestrians at each link

    T = int(step_duration / sample_t)
    link_dict = {}
    next_state_dict = {}
    for link_name, link_attribute in env.link.items():
        if link_name != "generation_link":
            link_dict[link_name] = {"link_list": link_attribute["id"], 
                                    "length": link_attribute["length"], "width": link_attribute["width"], 
                                    "n_ped": np.zeros(T), "density": np.zeros(T)}
        next_state_dict[link_name] = 0

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
                            next_state_dict[link_name] += 1
    next_state = list(next_state_dict.values())

    # 混雑度
    congestion_degree = 0
    for link_name, link_attribute in link_dict.items():
        congestion_degree += sum(link_attribute["density"] > 0.71) #.astype(int)

    if n_obj == 1:
        reward = -int(congestion_degree)
    # else:
    #     reward = []

    done = False

    return next_state, reward, done
    


if __name__ == '__main__':

    args = sys.argv

    path_to_gym = args[1]
    env_name = args[2]
    step = int(args[3])
    step_duration = int(args[4])
    sim_step = int(args[5])
    sim_dir = args[6] 
    agent_log_dir = args[7]

    import sys
    sys.path.append((path_to_gym+"envs/"))
    from two_routes import TwoRoutesEnv

    if env_name == "two_routes":
        env = TwoRoutesEnv()


    log = pd.read_csv(sim_dir + "/log/log_individual_pedestrians.csv")
    next_state_, reward, done = get_state_reward(env, log, step_duration, sim_step)

    gen = pd.read_csv(sim_dir + "/generation.csv")
    next_generation_pedestrian_number = gen[gen.step == (step+1)]["n_ped"].values[0]

    next_state = [int(next_generation_pedestrian_number)]
    next_state.extend(next_state_)

    with open(agent_log_dir + "history.json", "r") as f:
        history = json.load(f)

    history[str(step)]["reward"] = reward
    history[str(step)]["next_state"] = next_state
    history[str(step)]["done"] = done
    history[str(step+1)] = {"sim_step": sim_step, "state": next_state, "action": np.nan, "reward": np.nan, "next_state": np.nan, "done": np.nan}

    with open(agent_log_dir + "history.json", "w") as f:
        json.dump(history, f, indent=2)


    