import numpy as np
from scipy.signal import butter,filtfilt, find_peaks
from scipy.signal import lfilter
from scipy import stats
import matplotlib.pyplot as plt


def get_head_pos(times, h_tilt, v_tilt, display=False):

    # Remove -1000s
    filtered_v_tilt = []
    filtered_times = []
    for i in range(len(times)):
        if v_tilt[i] != -1000:
            filtered_v_tilt.append(v_tilt[i])
            filtered_times.append(times[i])
    times = filtered_times
    # filtered_ears = butter_bandpass_filter(ears, np.array([0.1, 10]), fs, 5)



    if display:
        plt.figure(3333)
        plt.plot(np.array(times), filtered_v_tilt, label='Vertical Tilt')
        plt.title(f'Head tilt')
        plt.xlabel('Time (seconds)')
        plt.ylabel('tilt')
        plt.legend()
        # plt.show()
    