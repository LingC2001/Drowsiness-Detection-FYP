from eeglib.helpers import CSVHelper
from eeglib.wrapper import Wrapper
import pandas as pd
import numpy as np
import neurokit2 as nk
import antropy as ent
from scipy import stats
import scipy
import pywt
from multiprocessing import Pool, Process
import os


def WE(y, level=4, wavelet='coif2'):
    from math import log
    n = len(y)

    sig = y

    ap = {}

    for lev in range(0, level):
        (y, cD) = pywt.dwt(y, wavelet)
        ap[lev] = y

    # Energy

    Enr = np.zeros(level)
    for lev in range(0, level):
        Enr[lev] = np.sum(np.power(ap[lev], 2)) / n

    Et = np.sum(Enr)

    Pi = np.zeros(level)
    for lev in range(0, level):
        Pi[lev] = Enr[lev] / Et

    we = - np.sum(np.dot(Pi, np.log(Pi)))

    return we


def extract_eeg(filename):
    folder_path = "DATA/bci_filtered"
    save_folder = "DATA/EEG_features_30s"
    fs = 125
    win_size = fs*30
    step_size = fs*20
    print(folder_path + '/' + filename)
    helper = CSVHelper(folder_path + '/' + filename, 
                    selectedSignals = [2,3,4], 
                    sampleRate = fs, 
                    normalize = True, 
                    ICA= True,
                    windowSize = win_size)

    helper.prepareIterator(step = step_size)
    wrapper = Wrapper(helper)

    wrapper.addFeature.bandPower()
    wrapper.addFeature.hjorthActivity()
    wrapper.addFeature.hjorthMobility()
    wrapper.addFeature.hjorthComplexity()
    wrapper.addFeature.sampEn()
    wrapper.addFeature.DFA()
    wrapper.addFeature.PFD()
    wrapper.addFeature.LZC()
    wrapper.addFeature.HFD()


    data_eeglib = wrapper.getAllFeatures()

    
    all_data = pd.read_csv(folder_path + '/' + filename, header= None)
    eeg_data = all_data.iloc[:,[2,3,4]]

    features = ["ApEnt", "SaEnt", "Sh_Ent",
                "fuzzy", "Multiscale", "Sp_ent", "wave_ent", "EEG_mean", "EEG_std",
                "EEG_kurt", "EEG_var", "EEG_skew"]
    df0 = pd.DataFrame(columns = [x+"0" for x in features])
    df1 = pd.DataFrame(columns = [x+"1" for x in features])
    df2 = pd.DataFrame(columns = [x+"2" for x in features])

    for c in range(3):
        for i in range(0, eeg_data.shape[0] - win_size + 1, step_size):
            data = eeg_data.iloc[i: i+win_size, c].T
            # print(data.shape)


            ApEnt0 = nk.entropy_approximate(data)[0]
            SaEnt0 = nk.entropy_sample(data)[0]
            Sh_Ent0 = nk.entropy_shannon(data)[0]
            fuzzy0 = nk.entropy_fuzzy(data)[0]
            Multiscale0 = nk.entropy_multiscale(data)[0]

            # Other ENT
            Sp_ent0 = ent.spectral_entropy(data, fs)
            wave_ent0 = WE(data)

            # OTHER FEATURES
            EEG_mean0 = np.nanmean(data)
            EEG_std0 = np.nanstd(data)
            EEG_kurt0 = stats.kurtosis(data, nan_policy='omit')
            EEG_var0 = np.nanvar(data)
            EEG_skew0 = stats.skew(data, nan_policy='omit')

            EEG_features_out0 = np.hstack((ApEnt0, SaEnt0, Sh_Ent0,
                                            fuzzy0, Multiscale0, Sp_ent0, wave_ent0, EEG_mean0, EEG_std0,
                                            EEG_kurt0, EEG_var0, EEG_skew0))
            if c == 0:
                df0.loc[len(df0)] = EEG_features_out0
            elif c == 1:
                df1.loc[len(df1)] = EEG_features_out0
            elif c == 2:
                df2.loc[len(df2)] = EEG_features_out0

    df = pd.concat([data_eeglib, df0,df1,df2], axis=1)
    df.to_csv(save_folder + '/' + filename[0:-13] + "_eeg_features.csv", index=False)
    return filename

if __name__ == '__main__':

    folder_path = "DATA/bci_filtered"
    filenames = []
    for f in os.listdir(folder_path):
        if f.endswith('.csv'):
            filenames.append(f)

    with Pool(processes=8) as pool:
        results = pool.imap_unordered(extract_eeg, filenames)
        for f in results:
            print("extracted" + f)