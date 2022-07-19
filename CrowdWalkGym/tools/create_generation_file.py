import os
import sys
import glob
import json
import datetime
import numpy as np
import pandas as pd
import datetime
format = '%Y:%m:%d-%H:%M:%S:%f'

args = sys.argv
csv_file_path = args[1]
output_file_path = args[2]
start_datetime = args[3]
data = pd.read_csv(csv_file_path)
data_num = list(data["n_ped"])


gen = []
for t, p in enumerate(data_num):
    time_base = datetime.datetime.strptime(start_datetime, '%H:%M:%S') + datetime.timedelta(seconds=t*30)
    for i in range(p):
        time = time_base + datetime.timedelta(seconds=np.random.randint(0, 30))
        time = datetime.datetime.strftime(time, '%H:%M:%S')
        agent = {"rule": "EACH", "agentType": {"className": "RubyAgent", "rubyAgentClass": "UtilityAgent"},
                "startTime": time, "total": 1, "duration": 1, "startPlace": "generation_link", 
                 "goal": "goal","conditions": []}
        gen.append(agent)
gen_each = sorted(gen, key=lambda x: x['startTime'])

gen_each_id = []
for i, r in enumerate(gen_each):
    agent = r.copy()
    agent["conditions"] = [str(i+1)+"_AGENT", agent["startTime"], None, None, None]
    gen_each_id.append(agent)
with open(output_file_path, "w") as f:
    json.dump(gen_each_id, f,  indent=2, ensure_ascii=False)

# add #{ "version" : 2}