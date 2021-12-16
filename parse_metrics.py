# %%
import os
import os.path
import json

workingdir = os.path.join(os.getcwd(), "output", "ANN")
print(workingdir)

metrics = {}

for dirpath, dirnames, filenames in os.walk(workingdir):
    # print(dirnames)
    for filename in [f for f in filenames if f.endswith(".json")]:
        dirpath_split = dirpath.split('\\')
        ann = dirpath_split.index('ANN')
        score_func = dirpath_split[ann + 1]
        activation = dirpath_split[ann + 2]

        name = filename.split('.json')[0]
        file_split = name.split('-')
        # activation = 
        if len(file_split) == 5:
            _, _, l1, l2, dropout = file_split
            l3 = l1
        elif len(file_split) == 6:
            _, _, l1, l2, l3, dropout = file_split
        else:
            raise Exception("filename not parsable")

        with open(os.path.join(dirpath, filename), "rb") as f:
            d = json.load(f)  
            metrics[score_func, activation, l1, l2, l3, dropout] = d
        pass


# %%
import numpy as np 
best_metrics = {}

for key, value in metrics.items():
    i = np.argmax(value["hr"])
    best_metrics[key] = (value["hr"][i], value["r"][i], value["ndcg"][i], i)

# %% 
from pprint import pprint
best_metrics_list = list(best_metrics.items())
s = sorted(best_metrics_list, key= lambda x: x[1][0])
pprint(s[-5:])
