import pandas as pd
import csv
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.signal import lfilter
from scipy import stats
from scipy.signal import butter,filtfilt, find_peaks
from utils.get_blinking import get_blinking

fs = 60
folder_path = "DATA/BULK"
with open(folder_path + "/M_A_N_vs_blinking_rate.csv", 'w', newline='') as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["Morning", "Afternoon", "Night"])
    files_to_analyse = []
    for f in os.listdir(folder_path):
        if f.endswith("morning.csv") or f.endswith("afternoon.csv") or f.endswith("night.csv"):   # change this later
            files_to_analyse.append(f)

    while files_to_analyse != []:
        curr_row_to_write = [-1]*3

        curr_file = files_to_analyse[0]
        files_to_analyse.pop(0)
        _count = 0
        for l in range(len(curr_file)):
            if curr_file[l] == '_':
                _count += 1
            
            if _count == 2:
                video_id = curr_file[0:l]
                break
        
        print("\nAnalysing " + curr_file)

        data = pd.read_csv(folder_path + '/' + curr_file)
        times = (data["Time(s)"].to_numpy()).tolist()
        ears = (data["EAR"].to_numpy()).tolist()

        # # Cropping data:
        # cropping_range = [0]*2
        # cropping_range[1] = len(times)- (60*fs)
        # cropping_range[0] = cropping_range- (30*60*fs)

        total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=False)

        if curr_file.endswith("morning.csv"):
            curr_row_to_write[0] = avg_blink_rate_per_min
        elif curr_file.endswith("afternoon.csv"):
            curr_row_to_write[1] = avg_blink_rate_per_min
        elif curr_file.endswith("night.csv"):
            curr_row_to_write[2] = avg_blink_rate_per_min
        #print(avg_blink_rate_per_min)

        print(video_id)
        file_idx_to_remove = []
        for file_idx in range(len(files_to_analyse)):
            if files_to_analyse[file_idx].startswith(video_id):
                curr_file = files_to_analyse[file_idx]
                file_idx_to_remove.append(file_idx)

                print("\nAnalysing " + curr_file)

                data = pd.read_csv(folder_path + '/' + curr_file)
                times = (data["Time(s)"].to_numpy()).tolist()
                ears = (data["EAR"].to_numpy()).tolist()
                
                total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=False)

                if curr_file.endswith("morning.csv"):
                    curr_row_to_write[0] = avg_blink_rate_per_min
                elif curr_file.endswith("afternoon.csv"):
                    curr_row_to_write[1] = avg_blink_rate_per_min
                elif curr_file.endswith("night.csv"):
                    curr_row_to_write[2] = avg_blink_rate_per_min

        files_to_analyse = [x for y, x in enumerate(files_to_analyse) if y not in file_idx_to_remove]
        writer.writerow(curr_row_to_write)
        print(files_to_analyse)







