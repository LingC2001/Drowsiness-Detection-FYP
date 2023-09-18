"""
File for compiling all the features with no grouping in 15 seconds intervals instead of 30 minutes with interpolation labelling
"""

import pandas as pd
import numpy as np
import os
from face_analysis.blinking.utils.get_blinking import get_blinking

class DataCompiler:
    def __init__(self):
        self.folder_path_pre = "../KSS_sorted_features/Pre"
        self.folder_path_post = "../KSS_sorted_features/Post"

        self.cap_files_pre = {}
        self.cap_files_post = {}


    def get_files(self):
        for f in os.listdir(self.folder_path_pre):
            for sub_folder in os.listdir(self.folder_path_pre + '/' + f):
                if sub_folder == "cap":
                    for sub_f in os.listdir(self.folder_path_pre + '/' + f + '/' + sub_folder):
                        if sub_f.endswith(".pkl"):
                            self.cap_files_pre[sub_f[0:-26]] = self.folder_path_pre + "/" + f + '/' + sub_folder + '/' + sub_f
        
        for f in os.listdir(self.folder_path_post):
            for sub_folder in os.listdir(self.folder_path_post + '/' + f):
                if sub_folder == "cap":
                    for sub_f in os.listdir(self.folder_path_post + '/' + f + '/' + sub_folder):
                        if sub_f.endswith(".pkl"):
                            self.cap_files_post[sub_f[0:-26]] = self.folder_path_post + "/" + f + '/' + sub_folder + '/' + sub_f
    
    def compile_all(self):
        fs = 60
        win_size = 15*fs
        step_size = 5*fs
        
        features_list = ["EAR_mean", "EAR_std", "blinking_rate", "blink_duration", "MAR_mean", "MAR_std", "PUC_mean", "PUC_std", "MOE_mean", "MOE_std", "vtilt_mean", "vtilt_std", "htilt_mean", "htilt_std"]
        
        for k in self.cap_files_pre:
            df = pd.read_pickle(self.cap_files_pre[k] ,compression="bz2")
            new_df = pd.DataFrame(columns=features_list)
            for i in range(0, df.shape[0] - win_size + 1, step_size):
                data = df.iloc[i: i+win_size, :]

                times = data["Time(s)"].to_numpy().tolist()
                
                ear_mean = np.mean(data["EAR"].to_numpy())
                ear_std = np.std(data["EAR"].to_numpy())

                total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time, blink_duration = get_blinking(times, data["EAR"].to_numpy().tolist(), display=False, get_blink_duration=True)

                mar_mean = np.mean(data["MAR"].to_numpy())
                mar_std = np.std(data["MAR"].to_numpy())

                puc_mean = np.mean(data["PUC"].to_numpy())
                puc_std = np.std(data["PUC"].to_numpy())

                moe_mean = np.mean(data["MOE"].to_numpy())
                moe_std = np.std(data["MOE"].to_numpy())

                vtilt_mean = np.mean(data["Vertical Tilt"].to_numpy())
                vtilt_std = np.std(data["Vertical Tilt"].to_numpy())

                htilt_mean = np.mean(data["Horizontal Tilt"].to_numpy())
                htilt_std = np.std(data["Horizontal Tilt"].to_numpy())

                features = [ear_mean, ear_std, avg_blink_rate_per_min, blink_duration, mar_mean, mar_std, puc_mean, puc_std, moe_mean, moe_std, vtilt_mean, vtilt_std, htilt_mean, htilt_std]
                new_df.loc[len(new_df)] = features

            new_df.to_csv(self.cap_files_pre[k][0:-4] + ".csv")

        for k in self.cap_files_post:
            df = pd.read_pickle(self.cap_files_post[k] ,compression="bz2")
            new_df = pd.DataFrame(columns=features_list)
            for i in range(0, df.shape[0] - win_size + 1, step_size):
                data = df.iloc[i: i+win_size, :]

                times = data["Time(s)"].to_numpy().tolist()
                
                ear_mean = np.mean(data["EAR"].to_numpy())
                ear_std = np.std(data["EAR"].to_numpy())

                total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time, blink_duration = get_blinking(times, data["EAR"].to_numpy().tolist(), display=False, get_blink_duration=True)

                mar_mean = np.mean(data["MAR"].to_numpy())
                mar_std = np.std(data["MAR"].to_numpy())

                puc_mean = np.mean(data["PUC"].to_numpy())
                puc_std = np.std(data["PUC"].to_numpy())

                moe_mean = np.mean(data["MOE"].to_numpy())
                moe_std = np.std(data["MOE"].to_numpy())

                vtilt_mean = np.mean(data["Vertical Tilt"].to_numpy())
                vtilt_std = np.std(data["Vertical Tilt"].to_numpy())

                htilt_mean = np.mean(data["Horizontal Tilt"].to_numpy())
                htilt_std = np.std(data["Horizontal Tilt"].to_numpy())

                features = [ear_mean, ear_std, avg_blink_rate_per_min, blink_duration, mar_mean, mar_std, puc_mean, puc_std, moe_mean, moe_std, vtilt_mean, vtilt_std, htilt_mean, htilt_std]
                new_df.loc[len(new_df)] = features

            new_df.to_csv(self.cap_files_post[k][0:-4] + ".csv")


    def run(self):
        self.get_files()
        self.compile_all()


if __name__ == "__main__":
    compiler = DataCompiler()
    compiler.run()