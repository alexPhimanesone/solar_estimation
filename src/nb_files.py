import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sequences_dir = "D:/DATA/BloomLive_Audio8661/sequences"

date0 = "2024-05-30"
start_timestamp = 80000
#end_timestamp   = 170000


def stats(date0):
    
    # Get all relevant directories
    list_dir = [dir for dir in os.listdir(sequences_dir)
                if dir[len("sequence_"):len("sequence_YYYY-MM-DD")] == date0]
    list_dir.sort()
    
    # Build a lists 
    list_timestamps = []
    list_file_numbers = []
    for i in range(len(list_dir)):
        timestamp = list_dir[i][len("sequence_YYYY-MM-DDT"):len("sequence_YYYY-MM-DDThhmmss")]
        if start_timestamp <= int(timestamp): # and int(timestamp) <= end_timestamp
            list_timestamps.append(timestamp)
            list_file_numbers.append(len(os.listdir((os.path.join(sequences_dir, list_dir[i])))))

    # Compute statistics
    mean   = np.mean(  list_file_numbers)
    std    = np.std(   list_file_numbers)
    median = np.median(list_file_numbers)

    # Plot graph
    plt.figure()
    plt.plot(list_timestamps, list_file_numbers)
    plt.xticks(list_timestamps[::max(1, len(list_timestamps)//20)], rotation=45, ha='right')
    plt.xlabel('Timestamps')
    plt.ylabel('Number of Files')
    plt.show()

    return mean, median, std, dict


mean, median, std, dict = stats(date0)
print(mean, median, std)