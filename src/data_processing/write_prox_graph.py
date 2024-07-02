import os
import numpy as np

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data"
dataset_dir  = os.path.join(data_dir   , "dataset/")
metadata_dir = os.path.join(dataset_dir, "metadata/")


nb_endroit = 20

endroits_toit_laas     = [0, 13]
endroits_derriere_laas = [1, 14]
endroits_devant_cnes   = [2, 3, 15, 16]
endroits_mare_arnaud   = [4, 5, 17]
endroits_derriere_cnes = [6, 7, 18, 19]
endroits_rucher        = [8]
endroits_maison        = [9, 10, 11]
endroits_terrain       = [12]

list_endroits = []
list_endroits.append(endroits_toit_laas)
list_endroits.append(endroits_derriere_laas)
list_endroits.append(endroits_devant_cnes)
list_endroits.append(endroits_mare_arnaud)
list_endroits.append(endroits_derriere_cnes)
list_endroits.append(endroits_rucher)
list_endroits.append(endroits_maison)
list_endroits.append(endroits_terrain)

prox_graph = np.zeros((nb_endroit, nb_endroit))
for i in range(nb_endroit):
    for j in range(nb_endroit):
        for endroits in list_endroits:
            if i in endroits and j in endroits:
                prox_graph[i, j] = 1
print(prox_graph)
np.save(os.path.join(metadata_dir, "prox_graph_fin.npy"), prox_graph)