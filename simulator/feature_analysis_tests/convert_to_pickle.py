import pandas as pd
import os
import tqdm

def convert(f):
    df = pd.read_pickle(folder_path + "/" + f, compression="bz2")
    df.to_csv(save_folder + "/" + f[0:-18] + "_emg_features_filtered.csv")
    # df = pd.read_pickle("DATA/test/test_landmarks" + ".pkl", compression='bz2')
    print("Converted " + f + " to pkl (compression = bz2)")

def cap_convert(f):
    df = pd.read_pickle(folder_path + "/" + f, compression="bz2")
    file_prefix = string_maker(f)
    df.to_pickle(save_folder + "/" + file_prefix + "_cap_features_filtered.pkl", compression='bz2')
    # df = pd.read_pickle("DATA/test/test_landmarks" + ".pkl", compression='bz2')
    print("Converted " + f + " to pkl (compression = bz2)")

def string_maker(f):
    if f[4]=="1" and not f[5] == "_":
        file_prefix = f[4] + f[5] + f[7]
    else:
        file_prefix = f[4] + f[6]
    return file_prefix

if __name__ == "__main__":
    filenames = []
    folder_path = "results_45s/emg_45_csv/"
    save_folder = "results_45s/emg_45_csv/"
    
    for f in os.listdir(folder_path):
        if f.endswith("filtered.pkl"):
            filenames.append(f)

    print(filenames)
    for f in filenames:
        convert(f)