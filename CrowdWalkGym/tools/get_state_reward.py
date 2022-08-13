# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import pandas as pd
import os
import json
import sys

def get_state_reward(env, log, step_duration, step, log_dir, n_obj, sample_t = 30):

    # リンク上の各時刻の歩行者数および密度
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

    # congestion_degree
    congestion_degree = 0
    for link_name, link_attribute in link_dict.items():
        congestion_degree += sum(link_attribute["density"] >= 0.72) #.astype(int)

    # travel distance
    if os.path.exists(log_dir + "/agent_dict.json"):
        with open(log_dir + "/agent_dict.json", "r") as f:
            agent_dict = json.load(f)
    else:
        agent_dict = {}

    agent_log = pd.Series(list(set(list(log.pedestrian_tag)))).apply(lambda x: x[1:-1].split(', '))

    travel_distance = 0
    for i in range(len(agent_log)):
        # if the agent has arrived
        route = np.nan 
        distance = np.nan
        
        if agent_log[i][-1] == "arrived":
            
            agent_id = agent_log[i][0]
            route_ = agent_log[i][2]

            if route_ == "route1":
                route = 1
                # distance = 1. - env.route1_length - 0.3 # two-routes: 1-0.4-0.3 = 0.3
                distance = 1. - env.route1_length - 0.45 # moji: 1-0.3-0.45=0.25
            elif route_ == "route2":
                route = 2
                # distance = 1. - env.route2_length - 0.3 # two-routes: 1-0.7-0.3 = 0
                distance = 1. - env.route2_length - 0.45 # moji: 1-0.55-0.45 = 0

            if agent_id not in agent_dict:
                travel_distance += distance
            agent_dict[agent_id] = {"state": "arrived", "route":route, "travel_distance": distance}

    # print(len(agent_dict))
    with open(log_dir + "/agent_dict.json", "w") as f:
        json.dump(agent_dict, f,  indent=2, ensure_ascii=False)


    # reward
    if n_obj == 1:
        reward = -int(congestion_degree)
    elif n_obj == 2:
        reward = [int(travel_distance), -int(congestion_degree)]

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
    n_obj = int(args[8])

    import sys
    sys.path.append((path_to_gym+"envs/"))
    from two_routes import TwoRoutesEnv
    from moji import MojiEnv, MojiSmallEnv

    if env_name == "two_routes":
        env = TwoRoutesEnv()
    elif env_name == "moji":
        env = MojiEnv()
    elif env_name == "moji_small":
        env = MojiSmallEnv()

    log = pd.read_csv(agent_log_dir + "/log_individual_pedestrians.csv")
    next_state_, reward, done = get_state_reward(env, log, step_duration, sim_step, agent_log_dir, n_obj)

    gen = pd.read_csv(sim_dir + "/generation.csv")
    next_generation_pedestrian_number = gen[gen.step == (step+1)]["n_ped"].values[0]

    next_state = [step+1, int(next_generation_pedestrian_number)]
    next_state.extend(next_state_)
    

    with open(agent_log_dir + "/history.json", "r") as f:
        history = json.load(f)

    history[str(step)]["reward"] = reward
    history[str(step)]["next_state"] = next_state
    history[str(step)]["done"] = done
    history[str(step+1)] = {"sim_step": sim_step, "state": next_state, "action": np.nan, "reward": np.nan, "next_state": np.nan, "done": np.nan}

    with open(agent_log_dir + "/history.json", "w") as f:
        json.dump(history, f, indent=2)


    
