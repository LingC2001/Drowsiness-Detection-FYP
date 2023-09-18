"""
File for compiling all the features with no grouping
"""

import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from face_analysis.blinking.utils.get_blinking import get_blinking

class DataCompiler:
    def __init__(self):
        self.folder_path = "../KSS_sorted_features/Post"  #Main folder containing the numbers

        self.cap_files = {
            "awake": [],
            "drowsy": []
        }

        self.eeg_files = {
            "awake": [],
            "drowsy": []
        }

        self.ecg_files = {
            "awake": [],
            "drowsy": []
        }

        self.emg_files = {
            "awake": [],
            "drowsy": []
        }

        self.cap_funcs = []
        self.cap_funcs_param = []
        self.eeg_funcs = []
        self.eeg_funcs_param = []
        self.ecg_funcs = []
        self.ecg_funcs_param = []
        self.emg_funcs = []
        self.emg_funcs_param = []

        self.df = pd.DataFrame()

    def get_files(self):
        for f in os.listdir(self.folder_path):
            if int(f) <= 6:
                for sub_folder in os.listdir(self.folder_path + '/' + f):
                    if sub_folder == "eeg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.eeg_files["awake"].append(self.folder_path + "/" + f + '/' + sub_folder + '/' + sub_f)
                    elif sub_folder == "emg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.emg_files["awake"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "ecg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.ecg_files["awake"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "cap":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.cap_files["awake"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
            elif int(f) >= 7:
                for sub_folder in os.listdir(self.folder_path + '/' + f):
                    if sub_folder == "eeg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.eeg_files["drowsy"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "emg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.emg_files["drowsy"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "ecg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.ecg_files["drowsy"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "cap":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.cap_files["drowsy"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
    
    def add_analysis(self, data, func, param=None):
        if data == "cap":
            self.cap_funcs.append(func)
            self.cap_funcs_param.append(param)
        elif data == "ecg":
            self.ecg_funcs.append(func)
            self.ecg_funcs_param.append(param)
        elif data == "eeg":
            self.eeg_funcs.append(func)
            self.eeg_funcs_param.append(param)
        elif data == "emg":
            self.emg_funcs.append(func)
            self.emg_funcs_param.append(param)    

    def analyse_all(self):
        for (func, param) in zip(self.cap_funcs, self.cap_funcs_param):
            func(self.cap_files, param)
        
        for (func, param) in zip(self.eeg_funcs, self.eeg_funcs_param):
            func(self.eeg_files, param)

        for (func, param) in zip(self.ecg_funcs, self.ecg_funcs_param):
            func(self.ecg_files, param)

        for (func, param) in zip(self.emg_funcs, self.emg_funcs_param):
            func(self.emg_files, param)
        

    def run(self):
        self.get_files()
        self.analyse_all()
        self.df.to_csv("features_analysis_all.csv", index=False)
    
    def blinking(self, files, none_param):
        # blinking

        awake_files = files["awake"]
        drowsy_files = files["drowsy"]

        all_blinking = []
        for i in range(len(awake_files)):
            print("analysing: " + awake_files[i])
            data = pd.read_pickle(awake_files[i], compression="bz2")
            times = (data["Time(s)"].to_numpy()).tolist()
            ears = (data["EAR"].to_numpy()).tolist()

            total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=False)
            all_blinking.append(avg_blink_rate_per_min)

        for i in range(len(drowsy_files)):
            print("analysing: " + drowsy_files[i])
            data = pd.read_pickle(drowsy_files[i], compression="bz2")
            times = (data["Time(s)"].to_numpy()).tolist()
            ears = (data["EAR"].to_numpy()).tolist()

            total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=False)
            all_blinking.append(avg_blink_rate_per_min)

        all_col = pd.DataFrame({"blinking": all_blinking})

        self.df = pd.concat([self.df, all_col], axis=1)


    def mean(self, files, column_name):

        print("Averaging column: " + str(column_name))
        awake_files = files["awake"]
        drowsy_files = files["drowsy"]
        
        all_vals = []
        # analyse awake data
        for i in range(len(awake_files)):
            temp_df = pd.read_pickle(awake_files[i], compression='bz2')
            column_data = temp_df[column_name].to_numpy()
            all_vals.append(np.mean(column_data))

        # analyse drowsy data
        for i in range(len(drowsy_files)):
            temp_df = pd.read_pickle(drowsy_files[i], compression="bz2")
            column_data = temp_df[column_name].to_numpy()
            all_vals.append(np.mean(column_data))

        all_col = pd.DataFrame({"avg_" + column_name: all_vals})

        self.df = pd.concat([self.df, all_col], axis=1)

    def std(self, files, column_name):
        print("Finding Std for column: " + str(column_name))

        awake_files = files["awake"]
        drowsy_files = files["drowsy"]
        
        all_vals = []
        # analyse awake data
        for i in range(len(awake_files)):
            temp_df = pd.read_pickle(awake_files[i], compression='bz2')
            column_data = temp_df[column_name].to_numpy()
            all_vals.append(np.std(column_data))

        # analyse drowsy data
        for i in range(len(drowsy_files)):
            temp_df = pd.read_pickle(drowsy_files[i], compression="bz2")
            column_data = temp_df[column_name].to_numpy()
            all_vals.append(np.std(column_data))

        all_col = pd.DataFrame({"std_" + column_name: all_vals})
        self.df = pd.concat([self.df, all_col], axis=1)


    def eeg_means(self, files, none_param):
        print("Averaging EEG bandpowers")
        awake_files = files["awake"]
        drowsy_files = files["drowsy"]
        all_files = [awake_files, drowsy_files]

        all_vals_delta = []

        all_vals_theta = []

        all_vals_alpha = []

        all_vals_beta = []

        all_vals_hjorth_activity = []

        all_vals_hjorth_mobility = []

        all_vals_hjorth_complexity = []

        all_vals_DFA = []

        all_vals_PFD = []

        all_vals_LZC = []

        all_vals_HFD = []

        all_vals_sample_entropy = []

        all_vals_fuzzy_entropy = []

        all_vals_spectral_entropy = []

        all_vals_wave_entropy = []

        for k in range(len(all_files)):
            # analyse awake data
            for i in range(len(all_files[k])):
                temp_df = pd.read_pickle(all_files[k][i], compression='bz2')

                # delta
                val = self._eeg_mean_aux(temp_df, "bandPower(){}_0_delta", "bandPower(){}_1_delta", "bandPower(){}_2_delta")
                if k == 0:
                    all_vals_delta.append(val)
                elif k ==1:
                    all_vals_delta.append(val)
                
                # theta
                val = self._eeg_mean_aux(temp_df, "bandPower(){}_0_theta", "bandPower(){}_1_theta", "bandPower(){}_2_theta")
                if k == 0:
                    all_vals_theta.append(val)
                elif k ==1:
                    all_vals_theta.append(val)

                # alpha
                val = self._eeg_mean_aux(temp_df, "bandPower(){}_0_alpha", "bandPower(){}_1_alpha", "bandPower(){}_2_alpha")
                if k == 0:
                    all_vals_alpha.append(val)
                elif k ==1:
                    all_vals_alpha.append(val)
                
                # beta
                val = self._eeg_mean_aux(temp_df, "bandPower(){}_0_beta", "bandPower(){}_1_beta", "bandPower(){}_2_beta")
                if k == 0:
                    all_vals_beta.append(val)
                elif k ==1:
                    all_vals_beta.append(val)

                #hjorth Activity
                val = self._eeg_mean_aux(temp_df, "hjorthActivity(){}_0", "hjorthActivity(){}_1", "hjorthActivity(){}_2")
                if k == 0:
                    all_vals_hjorth_activity.append(val)
                elif k ==1:
                    all_vals_hjorth_activity.append(val)
                
                #hjorth Mobility
                val = self._eeg_mean_aux(temp_df, "hjorthMobility(){}_0", "hjorthMobility(){}_1", "hjorthMobility(){}_2")
                if k == 0:
                    all_vals_hjorth_mobility.append(val)
                elif k ==1:
                    all_vals_hjorth_mobility.append(val)
                
                #hjorth Complexity
                val = self._eeg_mean_aux(temp_df, "hjorthComplexity(){}_0", "hjorthComplexity(){}_1", "hjorthComplexity(){}_2")
                if k == 0:
                    all_vals_hjorth_complexity.append(val)
                elif k ==1:
                    all_vals_hjorth_complexity.append(val)
                
                # DFA
                val = self._eeg_mean_aux(temp_df, "DFA(){}_0", "DFA(){}_1", "DFA(){}_2")
                if k == 0:
                    all_vals_DFA.append(val)
                elif k ==1:
                    all_vals_DFA.append(val)
                
                #PFD
                val = self._eeg_mean_aux(temp_df, "PFD(){}_0", "PFD(){}_1", "PFD(){}_2")
                if k == 0:
                    all_vals_PFD.append(val)
                elif k ==1:
                    all_vals_PFD.append(val)

                #LZC
                val = self._eeg_mean_aux(temp_df, "LZC(){}_0", "LZC(){}_1", "LZC(){}_2")
                if k == 0:
                    all_vals_LZC.append(val)
                elif k ==1:
                    all_vals_LZC.append(val)
                
                #HFD
                val = self._eeg_mean_aux(temp_df, "HFD(){}_0", "HFD(){}_1", "HFD(){}_2")
                if k == 0:
                    all_vals_HFD.append(val)
                elif k ==1:
                    all_vals_HFD.append(val)
                
                #sample_entropy
                val = self._eeg_mean_aux(temp_df, "sampEn(){}_0", "sampEn(){}_1", "sampEn(){}_2")
                if k == 0:
                    all_vals_sample_entropy.append(val)
                elif k ==1:
                    all_vals_sample_entropy.append(val)
                
                #fuzzy entropy
                val = self._eeg_mean_aux(temp_df, "fuzzy0", "fuzzy1", "fuzzy2")
                if k == 0:
                    all_vals_fuzzy_entropy.append(val)
                elif k ==1:
                    all_vals_fuzzy_entropy.append(val)
                
                #spectral entropy
                val = self._eeg_mean_aux(temp_df, "Sp_ent0", "Sp_ent1", "Sp_ent2")
                if k == 0:
                    all_vals_spectral_entropy.append(val)
                elif k ==1:
                    all_vals_spectral_entropy.append(val)

                #wave entropy
                val = self._eeg_mean_aux(temp_df, "wave_ent0", "wave_ent1", "wave_ent2")
                if k == 0:
                    all_vals_wave_entropy.append(val)
                elif k ==1:
                    all_vals_wave_entropy.append(val)




        all_col = pd.DataFrame({"avg_delta": all_vals_delta})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_theta": all_vals_theta})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_alpha": all_vals_alpha})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_beta": all_vals_beta})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_hjorth_activity": all_vals_hjorth_activity})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_hjorth_mobility": all_vals_hjorth_mobility})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_hjorth_complexity": all_vals_hjorth_complexity})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_DFA": all_vals_DFA})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_PFD": all_vals_PFD})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_LZC": all_vals_LZC})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_HFD": all_vals_HFD})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_sample_entropy": all_vals_sample_entropy})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_fuzzy_entropy": all_vals_fuzzy_entropy})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_spectral_entropy": all_vals_spectral_entropy})
        self.df = pd.concat([self.df, all_col], axis=1)

        all_col = pd.DataFrame({"avg_wave_entropy": all_vals_wave_entropy})
        self.df = pd.concat([self.df, all_col], axis=1)



    def _eeg_mean_aux(self, temp_df, ch1, ch2, ch3):
        column_data_0 = temp_df[ch1].to_numpy()
        column_data_1 = temp_df[ch2].to_numpy()
        column_data_2 = temp_df[ch3].to_numpy()
        mean0 = np.mean(column_data_0)
        mean1 = np.mean(column_data_1)
        mean2 = np.mean(column_data_2)
        return np.mean([mean0, mean1, mean2])


if __name__ == "__main__":

    ecg_feature_names = ["mean_hr", "std_hr", "cvsd", "nni_20", "nni_50", "pnni_20", "pnni_50", "lf", "hf", "lf_hf_ratio"]
    emg_feature_names = ['VAR', 'RMS', 'IEMG', 'MAV', 'LOG', 'WL', 'ACC', 'DASDV', 'ZC', "MNP", "TP", "MNF", "MDF", "PKF", "WENT"]
    cap_feature_names = ["EAR", "MAR", "PUC", "MOE", "Vertical Tilt", "Horizontal Tilt"]
    
    compiler = DataCompiler()
    
    compiler.add_analysis("cap", compiler.blinking, None)

    for feature in cap_feature_names:
        compiler.add_analysis("cap", compiler.mean, feature)
    
    for feature in cap_feature_names:
        compiler.add_analysis("cap", compiler.std, feature)

    for feature in ecg_feature_names:
        compiler.add_analysis("ecg", compiler.mean, feature)
    
    for feature in ecg_feature_names:
        compiler.add_analysis("ecg", compiler.std, feature)
    
    compiler.add_analysis("eeg", compiler.eeg_means, None)

    for feature in emg_feature_names:
        compiler.add_analysis("emg", compiler.mean, feature)

    for feature in emg_feature_names:
        compiler.add_analysis("emg", compiler.std, feature)

    compiler.run()