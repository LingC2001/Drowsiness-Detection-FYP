"""
File for compiling all the features with no grouping in 15 seconds intervals instead of 30 minutes with interpolation labelling
"""

import pandas as pd
import numpy as np
import os

class DataCompiler:
    def __init__(self):
        self.folder_path_pre = "KSS_sorted_features/Pre"
        self.folder_path_post = "KSS_sorted_features/Post"

        self.cap_files = {}
        self.eeg_files = {}
        self.ecg_files = {}
        self.emg_files = {}
        self.common_filenames = []

        self.df = pd.DataFrame()

    def get_files(self):
        for f in os.listdir(self.folder_path_pre):
            for sub_folder in os.listdir(self.folder_path_pre + '/' + f):
                if sub_folder == "eeg":
                    for sub_f in os.listdir(self.folder_path_pre + '/' + f + '/' + sub_folder):
                        self.eeg_files[sub_f[0:-26]] = {"kss":[int(f)], "path": self.folder_path_pre + "/" + f + '/' + sub_folder + '/' + sub_f}
                elif sub_folder == "emg":
                    for sub_f in os.listdir(self.folder_path_pre + '/' + f + '/' + sub_folder):
                        self.emg_files[sub_f[0:-26]] = {"kss":[int(f)], "path": self.folder_path_pre + "/" + f + '/' + sub_folder + '/' + sub_f}
                elif sub_folder == "ecg":
                    for sub_f in os.listdir(self.folder_path_pre + '/' + f + '/' + sub_folder):
                        self.ecg_files[sub_f[0:-26]] = {"kss":[int(f)], "path": self.folder_path_pre + "/" + f + '/' + sub_folder + '/' + sub_f}
                elif sub_folder == "cap":
                    for sub_f in os.listdir(self.folder_path_pre + '/' + f + '/' + sub_folder):
                        if sub_f.endswith(".csv"):
                            self.cap_files[sub_f[0:-26]] = {"kss":[int(f)], "path": self.folder_path_pre + "/" + f + '/' + sub_folder + '/' + sub_f}
        
        for f in os.listdir(self.folder_path_post):
            for sub_folder in os.listdir(self.folder_path_post + '/' + f):
                if sub_folder == "eeg":
                    for sub_f in os.listdir(self.folder_path_post + '/' + f + '/' + sub_folder):
                        try:
                            self.eeg_files[sub_f[0:-26]]["kss"].append(int(f))
                        except KeyError:
                            pass
                elif sub_folder == "emg":
                    for sub_f in os.listdir(self.folder_path_post + '/' + f + '/' + sub_folder):
                        try:
                            self.emg_files[sub_f[0:-26]]["kss"].append(int(f))
                        except KeyError:
                            pass
                elif sub_folder == "ecg":
                    for sub_f in os.listdir(self.folder_path_post + '/' + f + '/' + sub_folder):
                        try:
                            self.ecg_files[sub_f[0:-26]]["kss"].append(int(f))
                        except KeyError:
                            pass
                elif sub_folder == "cap":
                    for sub_f in os.listdir(self.folder_path_post + '/' + f + '/' + sub_folder):
                        if sub_f.endswith(".csv"):
                            try:
                                self.cap_files[sub_f[0:-26]]["kss"].append(int(f))
                            except KeyError:
                                pass
        
        for k in self.cap_files.keys():
            if (k in self.eeg_files.keys()) and (k in self.ecg_files.keys()) and ((k in self.emg_files.keys())):
                self.common_filenames.append(k)
            
    
    def compile_all(self):
        
        for k in self.common_filenames:
            eeg_df = pd.read_pickle(self.eeg_files[k]["path"] ,compression="bz2")
            eeg_df.drop(columns=eeg_df.columns[0], axis=1, inplace=True)

            ecg_df = pd.read_pickle(self.ecg_files[k]["path"] ,compression="bz2")
            ecg_df.drop(columns=ecg_df.columns[0], axis=1, inplace=True)

            emg_df = pd.read_pickle(self.emg_files[k]["path"] ,compression="bz2")

            cap_df = pd.read_csv(self.cap_files[k]["path"])
            cap_df.drop(columns=cap_df.columns[0], axis=1, inplace=True)

            lens = [len(eeg_df), len(ecg_df), len(emg_df), len(cap_df)]
            min_len = np.min(lens)
            lens_to_crop = [x-min_len for x in lens]
 
            for i in range(len(lens_to_crop)):
                if lens_to_crop[i] > 0:
                    if i == 0:
                        eeg_df.drop(eeg_df.tail(lens_to_crop[i]).index, inplace=True)
                    elif i == 1:
                        ecg_df.drop(ecg_df.tail(lens_to_crop[i]).index, inplace=True)
                    elif i == 2:
                        emg_df.drop(emg_df.tail(lens_to_crop[i]).index, inplace=True)
                    elif i == 3:
                        cap_df.drop(cap_df.tail(lens_to_crop[i]).index, inplace=True)

            kss = self.eeg_files[k]["kss"]
            kss_interpolated = np.linspace(start=kss[0], stop=kss[1], num=eeg_df.shape[0])
            kss_df = pd.DataFrame({"kss": kss_interpolated})

            all_df = pd.concat([eeg_df, ecg_df, emg_df, cap_df, kss_df], axis=1)
            
            self.df = pd.concat([self.df, all_df], axis=0)
        
        # some more filtering
        self.df.drop(columns= ["FR", "cvi"], inplace=True)
        self.df.dropna(subset= list(self.df.columns), inplace=True)

        print(self.df)

    def run(self):
        self.get_files()
        self.compile_all()
        self.df.to_csv("features_and_labels_15s.csv", index=False)   


if __name__ == "__main__":
    compiler = DataCompiler()
    compiler.run()