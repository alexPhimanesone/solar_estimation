import os
import numpy as np

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data"
dataset_dir  = os.path.join(data_dir   , "dataset/")
metadata_dir = os.path.join(dataset_dir, "metadata/")

prox_list = [(2, 3), (4, 5), (6, 7), (9, 10)]
nb_endroit = 11


prox_graph = np.zeros((nb_endroit, nb_endroit))
for endroit1, endroit2 in prox_list:
    prox_graph[endroit1, endroit2] = 1
    prox_graph[endroit2, endroit1] = 1
np.save(os.path.join(metadata_dir, "prox_graph.npy"), prox_graph)