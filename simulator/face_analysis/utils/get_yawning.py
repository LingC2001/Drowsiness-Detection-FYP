import numpy as np
from scipy.signal import butter,filtfilt, find_peaks
from scipy.signal import lfilter
from scipy import stats
import matplotlib.pyplot as plt

def get_yawning(times, mars, display=False):
    filtered_mars = mars

    # Finding MODE of data
    vals_to_mode = []
    for i in range(len(mars)):
        if mars[i] < 1 and mars[i] >-0.2:
            vals_to_mode.append(mars[i])

    vals,counts = np.unique(np.round(np.array(vals_to_mode), 4), return_counts=True)
    index = np.argmax(counts)
    mar_mode = vals[index]

    #print(ear_mode)
    for i in range(len(filtered_mars)):
        if filtered_mars[i] > mar_mode*1.5 or filtered_mars[i] <-0.2:
            filtered_mars[i] = mar_mode

    # find peaks
    filtered_mars = np.array(filtered_mars)
    peak_idx, _ = find_peaks(filtered_mars, prominence=0.2, distance=30)

    total_yawns = len(peak_idx)

    if display:
        plt.figure(3)
        plt.plot(np.array(times), filtered_mars, label='MAR')
        plt.plot(np.array(times)[peak_idx], filtered_mars[peak_idx], "x", label='Yawn')
        plt.title(f'Total yawns: {len(peak_idx)}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Mouth Aspect Ratio')
        plt.legend()
        # plt.show()
    
    return total_yawns
