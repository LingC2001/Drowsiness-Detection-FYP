from scipy.signal import butter,filtfilt, find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from scipy.signal import lfilter
from scipy import stats
from blinking.utils.get_blinking import get_blinking
from utils.get_yawning import get_yawning
from utils.get_head_pos import get_head_pos



def butter_bandpass_filter(data, cutoff, fs, order):
    '''
    Butterworth bandpass filter
    data: data to be filtered
    cutoff: (has to be a numpy array) the two cutoff frequencies for the bandpass filter
    fs: sampling rate
    order: order of the filter
    return: filtered data
    '''
    normal_cutoff = cutoff / (fs*0.5)
    b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)
    y = filtfilt(b, a, data)
    return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path",
                    type=str)
    args = parser.parse_args()
    file_name = args.file_path
    
    fs = 60

    data = pd.read_pickle(file_name, compression = "bz2")

    times = (data["Time(s)"].to_numpy()).tolist()
    crop_end = len(times)-60*60
    crop_front = 60*60

    times = times[crop_front:crop_end]
    ears = (data["EAR"].to_numpy()).tolist()[crop_front:crop_end]
    mars = (data["MAR"].to_numpy()).tolist()[crop_front:crop_end]
    pucs = (data["PUC"].to_numpy()).tolist()[crop_front:crop_end]
    moes = (data["MOE"].to_numpy()).tolist()[crop_front:crop_end]
    h_tilt = (data["Horizontal Tilt"].to_numpy()).tolist()[crop_front:crop_end]
    v_tilt = (data["Vertical Tilt"].to_numpy()).tolist()[crop_front:crop_end]

    total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=True)
    total_yawns = get_yawning(times, mars, display=True)
    get_head_pos = get_head_pos(times, h_tilt, v_tilt, display = True)
    plt.show()
    