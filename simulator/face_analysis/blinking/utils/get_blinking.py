import numpy as np
from scipy.signal import butter,filtfilt, find_peaks
from scipy.signal import lfilter
from scipy import stats
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from .peak_remover import clip_data, ewma_fb, remove_outliers

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_blinking(times, ears, display=False):
    fs = 60
    
    # Remove -1000s
    filtered_ears = []
    filtered_times = []
    for i in range(len(times)):
        if ears[i] != -1000:
            filtered_ears.append(ears[i])
            filtered_times.append(times[i])
    times = filtered_times
    # filtered_ears = butter_bandpass_filter(ears, np.array([0.1, 10]), fs, 5)

    # Detecting blinks
    # Finding MODE of data
    vals_to_mode = []
    for i in range(len(ears)):
        if ears[i] < 1 and ears[i] >-0.2:
            vals_to_mode.append(ears[i])

    vals,counts = np.unique(np.round(np.array(vals_to_mode), 4), return_counts=True)
    index = np.argmax(counts)
    ear_mode = vals[index]

    #print(ear_mode)
    for i in range(len(filtered_ears)):
        if filtered_ears[i] > ear_mode*1.5 or filtered_ears[i] <-0.2:
            filtered_ears[i] = ear_mode

    # find peaks
    filtered_ears = np.array(filtered_ears)
    peak_idx, _ = find_peaks(-filtered_ears, prominence=0.12, distance=20)

    # remove outlier
    ear_mean = np.mean(filtered_ears[peak_idx])
    ear_std = np.std(filtered_ears[peak_idx])
    print("EAR Mean: " + str(ear_mean) + " || EAR Std: " + str(ear_std)+ " || EAR Mode: " + str(ear_mode))
    corrected_peak_idx = []
    for i in range(len(peak_idx)):
        if filtered_ears[peak_idx[i]] < ear_mode-ear_std:
            corrected_peak_idx.append(peak_idx[i])
    peak_idx = corrected_peak_idx
    print(f'Total blinks: {len(peak_idx)}')


    # Calculating blink rate
    # Average blinking rate:
    total_blinks = len(peak_idx)
    avg_blink_rate_per_min = total_blinks/(len(times)/60/60)
    #print(avg_blink_rate_per_min)

    # Continuous Blinking rate
    blink_intervals = []
    blink_interval_time = []
    for i in range(len(peak_idx)-1):
        blink_intervals.append(peak_idx[i+1] - peak_idx[i])
        blink_interval_time.append(peak_idx[i+1])

    avg_filter_size = 30
    blinking_rate = []
    blinking_rate_time = []
    for i in range(len(blink_intervals)-(avg_filter_size-1)):
        blinking_rate.append(avg_filter_size / np.sum(blink_intervals[i:i+avg_filter_size])*fs)
        blinking_rate_time.append(blink_interval_time[i+avg_filter_size-1]/fs)

    # Calculate average EAR over time
    avg_window_size = 180
    avg_ear = running_mean(filtered_ears, avg_window_size)
    avg_ear_time = times[avg_window_size-1::]
    
    # Caluclating change in eye openess
    # Remove blinks from EARs
    eye_openess = np.copy(filtered_ears)
    window_size = 300
    for i in range(int(np.ceil(eye_openess.shape[0]/window_size))):
        if i == np.ceil(eye_openess.shape[0]/window_size):
            vals_to_mode = eye_openess[i*window_size : -1]
        else: 
            vals_to_mode = eye_openess[i*window_size : (i+1)*window_size]
        vals,counts = np.unique(np.round(np.array(vals_to_mode), 4), return_counts=True)
        index = np.argmax(counts)
        t_ear_mode = vals[index]
        
        for j in range(i*window_size, i*window_size + vals_to_mode.shape[0]):
            if eye_openess[j] > t_ear_mode + 0.025 or  eye_openess[j] < t_ear_mode - 0.025:
                eye_openess[j] = t_ear_mode

    eye_openess = eye_openess - ear_mode
    avg_window_size = 1000
    eye_openess = running_mean(eye_openess, avg_window_size)
    eye_openess_time = times[avg_window_size-1::]



    if display:
        # plot data
        plt.figure(1)
        plt.plot(np.array(times)/60, filtered_ears, label='EAR')
        plt.plot(np.array(times)[peak_idx]/60, filtered_ears[peak_idx], "x", label='Blink')
        plt.title(f'Blinking Detection, Total blinks: {len(peak_idx)}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Eye Aspect Ratio')
        plt.legend()

        plt.figure(11)
        plt.plot(np.array(blinking_rate_time)/60, blinking_rate)
        plt.title(f'Blinking rate over time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Blinks/sec')
        #plt.show()

        plt.figure(1111)
        plt.plot(np.array(avg_ear_time)/60, avg_ear)
        plt.title(f'Mean EAR over time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Mean EAR')

        plt.figure(111)
        plt.plot(np.array(eye_openess_time)/60, eye_openess)
        plt.title(f'Eye openess')
        plt.xlabel('Time (minutes)')
        plt.ylabel('EAR')
        #plt.show()

        

    return total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time