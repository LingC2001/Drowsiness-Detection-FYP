from __future__ import division
import numpy as np
#from pyhrv.hrv import hrv
from ecgdetectors import Detectors
from ecg_functions import *
import pandas as pd
import hrvanalysis
import os
import math
from hrv.filters import quotient




def featureExtraction(rri, ECG, fs):
    time_domain = hrvanalysis.get_time_domain_features(rri)
    frequency_domain = hrvanalysis.get_frequency_domain_features(rri)
    other = hrvanalysis.get_csi_cvi_features(rri)

    HRV = [time_domain['mean_nni'], time_domain['cvnni']]
    Time_Features = [time_domain['cvsd'], time_domain['max_hr'], time_domain['mean_hr'], time_domain['median_nni'], time_domain['min_hr'], time_domain['nni_20'], time_domain['nni_50'], time_domain['pnni_20'],
                                                    time_domain['pnni_50'], time_domain['range_nni'], time_domain['rmssd'], time_domain['sdnn'], time_domain['sdsd'], time_domain['std_hr']]

    Frequency_Features = [frequency_domain['lf'], frequency_domain['hf'], frequency_domain['lf_hf_ratio'], frequency_domain['lfnu'], frequency_domain['hfnu'], frequency_domain['total_power'], frequency_domain['vlf']]
    Other_Features = [other['csi'], other['cvi']]

    return np.float64(np.concatenate((np.array(HRV), np.array(Time_Features), np.array(Frequency_Features), np.array(Other_Features))))

## CODE TO DETECT FEATURES 
fs = 125
wind_size = 30
step_size = 20 * fs
wind_n_elements = fs * wind_size
# ecg_features_list = ["HR", "VLF", "LF", "HF", "LF_HF", "Power", "rsp", "RRV_median", "RRV_mean", "RRV_ApEn", "HR_std", "Sh_Ent",
#                                 "Approx", "fuzzy", "wave_ent", "HRV_mean", "HRV_std", "HRV_kurt", "HRV_var", "HRV_skew"]
    
# detector = Detectors(fs)
ecg_features_list = ["mean_nni", "cvnni", "cvsd", "max_hr", "mean_hr", "median_nni", "min_hr", "nni_20", "nni_50", "pnni_20", "pnni_50", "range_nni",
                              "rmssd", "fuzzy", "sdnn", "sdsd", "std_hr", "lf", "hf", "lf_hf_ratio","lfnu", "total_power", "vlf","csi","cvi"]

# folder path
dir_path = "bci_filtered"
    
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        df = pd.DataFrame(columns=ecg_features_list)

        file_dir = "bci_filtered\\"+path
        data = np.array(pd.read_csv(file_dir, header=None))

        # data = np.array(pd.read_csv(file_dir))
        ecg_data = data[:,0]

        n_seconds_valid = np.floor(len(ecg_data)/fs)

        ecg_data_clipped = ecg_data[0:int(n_seconds_valid)*fs]

        features_array = np.zeros((math.ceil((len(ecg_data_clipped)-wind_n_elements+1)/(step_size)), 25))
        print(np.shape(features_array))
        
        # flag = not (n_windows == n_windows_check)
        
        dest_file_dir = "results_30s\\ecg_30_csv\\"+path
        counter = 0
        for i in range(0, len(ecg_data_clipped)-wind_n_elements+1, step_size):
            # print(i)
            # ecg_sample = ecg_data_clipped[i*wind_n_elements:i*wind_n_elements+wind_n_elements]
            ecg_sample = ecg_data_clipped[i: i+wind_n_elements]
            # print(len(ecg_sample))
            detector = Detectors(fs)
            rr_peaks = np.array(detector.pan_tompkins_detector(ecg_sample))
            # print(rr_peaks)

            rri = (np.diff(rr_peaks) / fs) * 10 ** 3
            filt_rri = np.array(quotient(rri))
            d = np.std(filt_rri)
            b0 = np.median(filt_rri)
            filt_rri[np.where(filt_rri > 1200)] = b0
            filt_rri[np.where(filt_rri < 600)] = b0

            try:
                features = featureExtraction(rri= filt_rri, ECG= ecg_sample, fs= fs)
                features_array[counter,:] += features
                counter += 1
            except ZeroDivisionError:
                counter += 1

        # print(np.shape(features_array))
        for j in range(np.shape(features_array)[0]):
            df.loc[len(df)] = features_array[j,:]

        df.to_csv(dest_file_dir[0:-13]+"_ecg_features.csv", index=False)    



