import scipy.signal as signal
import pandas as pd
import os
from multiprocessing import Pool, Process


def filter_bci(filename):
    folder_name = "DATA/bci_files/"
    save_folder = "DATA/bci_filtered/"
    
    f = filename

    df = pd.read_csv(folder_name + f)

    if f == "13m.csv":
        ecg_data = df.loc[:,"EXG Channel 4"]
        emg_data = df.loc[:,"EXG Channel 0"]
    else:
        ecg_data = df.loc[:,"EXG Channel 0"]
        emg_data = df.loc[:,"EXG Channel 4"]
    eeg0_data = df.loc[:,"EXG Channel 8"]
    eeg1_data = df.loc[:,"EXG Channel 9"]
    eeg2_data = df.loc[:,"EXG Channel 10"]

    order = 8
    fs = 125
    low_pass = signal.butter(order, 35, 'lowpass', fs=fs, output='sos')
    high_pass = signal.butter(order, 0.5, 'highpass', fs=fs, output='sos')
    notch1 = signal.butter(order, [48, 52], 'bandstop', fs=fs, output='sos')
    notch2 = signal.butter(order, [58, 62], 'bandstop', fs=fs, output='sos')


    filtered_ecg_data = signal.sosfilt(low_pass, ecg_data)
    filtered_ecg_data = signal.sosfilt(high_pass, filtered_ecg_data)
    filtered_ecg_data = signal.sosfilt(notch1, filtered_ecg_data)
    filtered_ecg_data = signal.sosfilt(notch2, filtered_ecg_data)
    filtered_ecg_data = pd.DataFrame(filtered_ecg_data)

    filtered_emg_data = signal.sosfilt(low_pass, emg_data)
    filtered_emg_data = signal.sosfilt(high_pass, filtered_emg_data)
    filtered_emg_data = signal.sosfilt(notch1, filtered_emg_data)
    filtered_emg_data = signal.sosfilt(notch2, filtered_emg_data)
    filtered_emg_data = pd.DataFrame(filtered_emg_data)

    filtered_eeg0_data = signal.sosfilt(low_pass, eeg0_data)
    filtered_eeg0_data = signal.sosfilt(high_pass, filtered_eeg0_data)
    filtered_eeg0_data = signal.sosfilt(notch1, filtered_eeg0_data)
    filtered_eeg0_data = signal.sosfilt(notch2, filtered_eeg0_data)
    filtered_eeg0_data = pd.DataFrame(filtered_eeg0_data)

    filtered_eeg1_data = signal.sosfilt(low_pass, eeg1_data)
    filtered_eeg1_data = signal.sosfilt(high_pass, filtered_eeg1_data)
    filtered_eeg1_data = signal.sosfilt(notch1, filtered_eeg1_data)
    filtered_eeg1_data = signal.sosfilt(notch2, filtered_eeg1_data)
    filtered_eeg1_data = pd.DataFrame(filtered_eeg1_data)

    filtered_eeg2_data = signal.sosfilt(low_pass, eeg2_data)
    filtered_eeg2_data = signal.sosfilt(high_pass, filtered_eeg2_data)
    filtered_eeg2_data = signal.sosfilt(notch1, filtered_eeg2_data)
    filtered_eeg2_data = signal.sosfilt(notch2, filtered_eeg2_data)
    filtered_eeg2_data = pd.DataFrame(filtered_eeg2_data)

    df_out = pd.concat([filtered_ecg_data, filtered_emg_data,filtered_eeg0_data,filtered_eeg1_data,filtered_eeg2_data], axis=1)

    df_out.to_csv(save_folder + '/' + f[0:-4] + "_filtered.csv", index=False, header=False)
    
    return filename


if __name__ == '__main__':

    folder_path = "DATA/bci_files"
    filenames = []
    for f in os.listdir(folder_path):
        if f.endswith('.csv'):
            filenames.append(f)

    print(filenames)

    with Pool(processes=8) as pool:
        results = pool.imap_unordered(filter_bci, filenames)
        for f in results:
            print("filtered" + f)