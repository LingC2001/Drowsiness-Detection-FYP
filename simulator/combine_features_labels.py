"""
File for compiling all the features with no grouping in X seconds intervals instead of 30 minutes with interpolation labelling
"""

import pandas as pd
import numpy as np
import os

class DataCompiler:
    def __init__(self):
        self.folder_path_pre = "KSS_sorted_15/Pre"
        self.folder_path_post = "KSS_sorted_15/Post"

        self.cap_files = {}
        self.eeg_files = {}
        self.ecg_files = {}
        self.emg_files = {}
        self.common_filenames = []

        self.df = pd.DataFrame()
        self.df_train = pd.DataFrame()
        self.df_valid = pd.DataFrame()
        self.df_test = pd.DataFrame()

    def get_files(self):
        for f in os.listdir(self.folder_path_pre):
            for sub_folder in os.listdir(self.folder_path_pre + '/' + f):
                if sub_folder == "eeg":
                    for sub_f in os.listdir(self.folder_path_pre + '/' + f + '/' + sub_folder):
                        self.eeg_files[sub_f[0:-17]] = {"kss":[int(f)], "path": self.folder_path_pre + "/" + f + '/' + sub_folder + '/' + sub_f}
                elif sub_folder == "emg":
                    for sub_f in os.listdir(self.folder_path_pre + '/' + f + '/' + sub_folder):
                        self.emg_files[sub_f[0:-17]] = {"kss":[int(f)], "path": self.folder_path_pre + "/" + f + '/' + sub_folder + '/' + sub_f}
                elif sub_folder == "ecg":
                    for sub_f in os.listdir(self.folder_path_pre + '/' + f + '/' + sub_folder):
                        self.ecg_files[sub_f[0:-17]] = {"kss":[int(f)], "path": self.folder_path_pre + "/" + f + '/' + sub_folder + '/' + sub_f}
                elif sub_folder == "cap":
                    for sub_f in os.listdir(self.folder_path_pre + '/' + f + '/' + sub_folder):
                        if sub_f.endswith(".csv"):
                            self.cap_files[sub_f[0:-17]] = {"kss":[int(f)], "path": self.folder_path_pre + "/" + f + '/' + sub_folder + '/' + sub_f}
        
        for f in os.listdir(self.folder_path_post):
            for sub_folder in os.listdir(self.folder_path_post + '/' + f):
                if sub_folder == "eeg":
                    for sub_f in os.listdir(self.folder_path_post + '/' + f + '/' + sub_folder):
                        try:
                            self.eeg_files[sub_f[0:-17]]["kss"].append(int(f))
                        except KeyError:
                            pass
                elif sub_folder == "emg":
                    for sub_f in os.listdir(self.folder_path_post + '/' + f + '/' + sub_folder):
                        try:
                            self.emg_files[sub_f[0:-17]]["kss"].append(int(f))
                        except KeyError:
                            pass
                elif sub_folder == "ecg":
                    for sub_f in os.listdir(self.folder_path_post + '/' + f + '/' + sub_folder):
                        try:
                            self.ecg_files[sub_f[0:-17]]["kss"].append(int(f))
                        except KeyError:
                            pass
                elif sub_folder == "cap":
                    for sub_f in os.listdir(self.folder_path_post + '/' + f + '/' + sub_folder):
                        if sub_f.endswith(".csv"):
                            try:
                                self.cap_files[sub_f[0:-17]]["kss"].append(int(f))
                            except KeyError:
                                pass
        for k in self.cap_files.keys():
            if (k in self.eeg_files.keys()) and (k in self.ecg_files.keys()) and ((k in self.emg_files.keys())):
                self.common_filenames.append(k)
        print(self.common_filenames)
    
    def compile_all(self):
        for k in self.common_filenames:
            # eeg_df = pd.read_pickle(self.eeg_files[k]["path"] ,compression="bz2")
            eeg_df = pd.read_csv(self.eeg_files[k]["path"])
            
            # ecg_df = pd.read_pickle(self.ecg_files[k]["path"] ,compression="bz2")
            ecg_df = pd.read_csv(self.ecg_files[k]["path"])

            # emg_df = pd.read_pickle(self.emg_files[k]["path"] ,compression="bz2")
            emg_df = pd.read_csv(self.emg_files[k]["path"])

            cap_df = pd.read_csv(self.cap_files[k]["path"])

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

            if self.eeg_files[k]["path"][-18] == "a":
                time_of_day = "afternoon" 
            elif self.eeg_files[k]["path"][-18] == "m":
                time_of_day = "morning"
            elif self.eeg_files[k]["path"][-18] == "n":
                time_of_day = "night"

            participant_id = [k[0:-1]] * eeg_df.shape[0]
            id_df = pd.DataFrame({"id": participant_id})

            time_of_day = [time_of_day] * eeg_df.shape[0]
            time_df = pd.DataFrame({"time_of_day": time_of_day})

            time_in_rec = np.linspace(start=0, stop=29.5, num=eeg_df.shape[0])
            time_rec_df = pd.DataFrame({"time_in_session": time_in_rec})

            kss = self.eeg_files[k]["kss"]
            kss_interpolated = np.linspace(start=kss[0], stop=kss[1], num=eeg_df.shape[0])
            kss_df = pd.DataFrame({"kss": kss_interpolated})

            all_df = pd.concat([eeg_df, ecg_df, emg_df, cap_df, time_rec_df, time_df, id_df, kss_df], axis=1)
            
            if k in ["2m", "2a", "2n", "6m", "6a", "6n", "8a", "8m", "8n"]:
                self.df_test = pd.concat([self.df_test, all_df], axis=0)
            elif k in ["3m", "3a", "3n", "4m", "4a", "4n", "14m", "14a", "14n"]:
                self.df_valid = pd.concat([self.df_valid, all_df], axis=0)
            else:
                self.df_train = pd.concat([self.df_train, all_df], axis=0)
            self.df = pd.concat([self.df, all_df], axis=0)
        
        # some more filtering
        self.df.drop(columns= ["FR", "cvi"], inplace=True)
        self.df.dropna(subset= list(self.df.columns), inplace=True)

        self.df_train.drop(columns= ["FR", "cvi"], inplace=True)
        self.df_train.dropna(subset= list(self.df_train.columns), inplace=True)

        self.df_valid.drop(columns= ["FR", "cvi"], inplace=True)
        self.df_valid.dropna(subset= list(self.df_valid.columns), inplace=True)

        self.df_test.drop(columns= ["FR", "cvi"], inplace=True)
        self.df_test.dropna(subset= list(self.df_test.columns), inplace=True)

        print(self.df)

    def run(self):
        self.get_files()
        self.compile_all()
        self.df.to_csv("features_and_labels_15s.csv", index=False)  
        self.df_train.to_csv("train_features_and_labels_15s.csv", index=False)  
        self.df_valid.to_csv("valid_features_and_labels_15s.csv", index=False)  
        self.df_test.to_csv("test_features_and_labels_15s.csv", index=False)   



if __name__ == "__main__":
    compiler = DataCompiler()
    compiler.run()