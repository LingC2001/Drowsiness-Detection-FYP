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
folder_path = "DATA/KSS_SORTED"



awake_KSS_files = []
drowsy_KSS_files = []
for f in os.listdir(folder_path):
    if int(f) <= 6:
        for sub_f in os.listdir(folder_path + '/' + f):
            if sub_f.endswith("features.pkl"):
                awake_KSS_files.append(folder_path + "/" + f + '/' + sub_f)
    elif int(f) >= 7:
        for sub_f in os.listdir(folder_path + '/' + f):
            if sub_f.endswith("features.pkl"):
                drowsy_KSS_files.append(folder_path + "/" + f + '/' + sub_f)

awake_blinking = []
for i in range(len(awake_KSS_files)):
    print("analysing: " + awake_KSS_files[i])
    data = pd.read_pickle(awake_KSS_files[i], compression="bz2")
    times = (data["Time(s)"].to_numpy()).tolist()
    ears = (data["EAR"].to_numpy()).tolist()

    total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=False)
    awake_blinking.append(avg_blink_rate_per_min)

drwosy_blinking = []
for i in range(len(drowsy_KSS_files)):
    print("analysing: " + drowsy_KSS_files[i])
    data = pd.read_pickle(drowsy_KSS_files[i], compression="bz2")
    times = (data["Time(s)"].to_numpy()).tolist()
    ears = (data["EAR"].to_numpy()).tolist()

    total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=False)
    drwosy_blinking.append(avg_blink_rate_per_min)


if len(awake_blinking) > len(drwosy_blinking):
    drwosy_blinking = drwosy_blinking + [np.nan]*(len(awake_blinking) -len(drwosy_blinking))
elif len(drwosy_blinking) > len(awake_blinking):
    awake_blinking = awake_blinking + [np.nan]*(len(drwosy_blinking) -len(awake_blinking))

with open("KSS_vs_blinking_rate.csv", 'w', newline='') as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["KSS<=6", "KSS>=7"])
    for i in range(len(drwosy_blinking)):
        writer.writerow([awake_blinking[i], drwosy_blinking[i]])
