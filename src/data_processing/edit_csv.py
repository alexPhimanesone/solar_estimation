import os
import csv

data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))
dataset_dir = os.path.join(data_dir, "dataset/")
metadata_dir = os.path.join(dataset_dir, "metadata/")


csv_path = os.path.join(metadata_dir, "endroits_metadata.csv")
fields = ["id_endroit", "id_pprad", "id_phone", "id_lens", "inclinaison", "timelapse"]

data = [
    {fields[0]: 0, fields[1]: 0, fields[2]: 0, fields[3]: 0, fields[4]: "(0, 0)", fields[5]: 1},   
]

# Write data to CSV file
with open(csv_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    
    # Write header
    writer.writeheader()
    
    # Write data rows
    for row in data:
        writer.writerow(row)

'''
entry = {}
entry[fields[0]] = 0
entry[fields[1]] = 0   
entry[fields[2]] = 0   
entry[fields[3]] = 0   
entry[fields[4]] = 0,0 
entry[fields[5]] = 1   

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    #writer.writerow(fields)
    writer.writerow([entry[fields[0]],
                     entry[fields[1]],
                     entry[fields[2]],
                     entry[fields[3]],
                     entry[fields[4]],
                     entry[fields[5]]])
'''