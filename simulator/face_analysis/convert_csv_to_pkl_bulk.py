import pandas as pd
import os
import tqdm

def convert(f):
    df = pd.read_csv(folder_path + "/" + f)
    df.to_pickle(save_folder + "/" + f[0:-4] + ".pkl", compression='bz2')
    # df = pd.read_pickle("DATA/test/test_landmarks" + ".pkl", compression='bz2')
    print("Converted " + f + " to pkl (compression = bz2)")

if __name__ == "__main__":
    filenames = []
    folder_path = "DATA/BULK"
    save_folder = "DATA/BULK_PKL_BZ2"
    
    for f in os.listdir(folder_path):
        if f.endswith("landmarks.csv") or f.endswith("features.csv"):
            filenames.append(f)

    print(filenames)
    for f in filenames:
        convert(f)