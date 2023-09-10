import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from face_analysis.blinking.utils.get_blinking import get_blinking

class DataCompiler:
    def __init__(self):
        self.folder_path = "Time_sorted_features"  #Main folder containing the numbers

        self.cap_files = {
            "morning": [],
            "afternoon": [],
            "night": []
        }

        self.ecg_files = {
            "morning": [],
            "afternoon": [],
            "night": []
        }

        self.eeg_files = {
            "morning": [],
            "afternoon": [],
            "night": []
        }

        self.emg_files = {
            "morning": [],
            "afternoon": [],
            "night": []
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
            if f == "morning":
                for sub_folder in os.listdir(self.folder_path + '/' + f):
                    if sub_folder == "eeg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.eeg_files["morning"].append(self.folder_path + "/" + f + '/' + sub_folder + '/' + sub_f)
                    elif sub_folder == "emg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.emg_files["morning"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "ecg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.ecg_files["morning"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "cap":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.cap_files["morning"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
            elif f == "afternoon":
                for sub_folder in os.listdir(self.folder_path + '/' + f):
                    if sub_folder == "eeg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.eeg_files["afternoon"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "emg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.emg_files["afternoon"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "ecg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.ecg_files["afternoon"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "cap":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.cap_files["afternoon"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
            elif f == "night":
                for sub_folder in os.listdir(self.folder_path + '/' + f):
                    if sub_folder == "eeg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.eeg_files["night"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "emg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.emg_files["night"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "ecg":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.ecg_files["night"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
                    elif sub_folder == "cap":
                        for sub_f in os.listdir(self.folder_path + '/' + f + '/' + sub_folder):
                            self.cap_files["night"].append(self.folder_path + "/" + f + '/' + sub_folder+ '/' + sub_f)
    
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
        self.df.to_csv("features_analysis_time.csv", index=False)
    
    def blinking(self, files, none_param):
        # blinking

        m_files = files["morning"]
        a_files = files["afternoon"]
        n_files = files["night"]

        morning_blinking = []
        for i in range(len(m_files)):
            print("analysing: " + m_files[i])
            data = pd.read_pickle(m_files[i], compression="bz2")
            times = (data["Time(s)"].to_numpy()).tolist()
            ears = (data["EAR"].to_numpy()).tolist()

            total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=False)
            morning_blinking.append(avg_blink_rate_per_min)

        afternoon_blinking = []
        for i in range(len(a_files)):
            print("analysing: " + a_files[i])
            data = pd.read_pickle(a_files[i], compression="bz2")
            times = (data["Time(s)"].to_numpy()).tolist()
            ears = (data["EAR"].to_numpy()).tolist()

            total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=False)
            afternoon_blinking.append(avg_blink_rate_per_min)
        
        night_blinking = []
        for i in range(len(n_files)):
            print("analysing: " + n_files[i])
            data = pd.read_pickle(n_files[i], compression="bz2")
            times = (data["Time(s)"].to_numpy()).tolist()
            ears = (data["EAR"].to_numpy()).tolist()

            total_blinks, avg_blink_rate_per_min, blinking_rate, blinking_rate_time = get_blinking(times, ears, display=False)
            night_blinking.append(avg_blink_rate_per_min)


        idx = np.argmax([len(morning_blinking), len(afternoon_blinking), len(night_blinking)])

        if idx == 0:
            afternoon_blinking = afternoon_blinking + [np.nan]*(len(morning_blinking) -len(afternoon_blinking))
            night_blinking = night_blinking + [np.nan]*(len(morning_blinking) -len(night_blinking))
        elif idx == 1:
            morning_blinking = morning_blinking + [np.nan]*(len(afternoon_blinking) -len(morning_blinking))
            night_blinking = night_blinking + [np.nan]*(len(afternoon_blinking) -len(night_blinking))
        elif idx == 2:
            morning_blinking = morning_blinking + [np.nan]*(len(night_blinking) -len(morning_blinking))
            afternoon_blinking = afternoon_blinking + [np.nan]*(len(night_blinking) -len(afternoon_blinking))

        m_col = pd.DataFrame({"blinking_morning": morning_blinking})
        a_col = pd.DataFrame({"blinking_afternoon": afternoon_blinking})
        n_col = pd.DataFrame({"blinking_night": night_blinking})

        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)


    def mean(self, files, column_name):
 
        print("Averaging column: " + str(column_name))
        m_files = files["morning"]
        a_files = files["afternoon"]
        n_files = files["night"]
        
        m_vals = []
        # analyse morning data
        for i in range(len(m_files)):
            temp_df = pd.read_pickle(m_files[i], compression='bz2')
            column_data = temp_df[column_name].to_numpy()
            m_vals.append(np.mean(column_data))

        a_vals = []
        # analyse afternoon data
        for i in range(len(a_files)):
            temp_df = pd.read_pickle(a_files[i], compression="bz2")
            column_data = temp_df[column_name].to_numpy()
            a_vals.append(np.mean(column_data))
        
        n_vals = []
        # analyse night data
        for i in range(len(n_files)):
            temp_df = pd.read_pickle(n_files[i], compression="bz2")
            column_data = temp_df[column_name].to_numpy()
            n_vals.append(np.mean(column_data))

        m_col = pd.DataFrame({"avg_" + column_name + "_morning": m_vals})
        a_col = pd.DataFrame({"avg_" + column_name + "_afternoon": a_vals})
        n_col = pd.DataFrame({"avg_" + column_name + "_night": n_vals})

        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

    def std(self, files, column_name):
        print("Finding Std for column: " + str(column_name))

        m_files = files["morning"]
        a_files = files["afternoon"]
        n_files = files["night"]
        
        m_vals = []
        # analyse m data
        for i in range(len(m_files)):
            temp_df = pd.read_pickle(m_files[i], compression='bz2')
            column_data = temp_df[column_name].to_numpy()
            m_vals.append(np.std(column_data))

        a_vals = []
        # analyse a data
        for i in range(len(a_files)):
            temp_df = pd.read_pickle(a_files[i], compression="bz2")
            column_data = temp_df[column_name].to_numpy()
            a_vals.append(np.std(column_data))
        
        n_vals = []
        # analyse a data
        for i in range(len(n_files)):
            temp_df = pd.read_pickle(n_files[i], compression="bz2")
            column_data = temp_df[column_name].to_numpy()
            n_vals.append(np.std(column_data))

        m_col = pd.DataFrame({"std_" + column_name + "_morning": m_vals})
        a_col = pd.DataFrame({"std_" + column_name + "_afternoon": a_vals})
        n_col = pd.DataFrame({"std_" + column_name + "_night": n_vals})

        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)


    def eeg_means(self, files, none_param):
        print("Averaging EEG bandpowers")

        m_files = files["morning"]
        a_files = files["afternoon"]
        n_files = files["night"]

        all_files = [m_files, a_files, n_files]

        m_vals_delta = []
        a_vals_delta = []
        n_vals_delta = []

        m_vals_theta = []
        a_vals_theta = []
        n_vals_theta = []

        m_vals_alpha = []
        a_vals_alpha = []
        n_vals_alpha = []

        m_vals_beta = []
        a_vals_beta = []
        n_vals_beta = []

        m_vals_hjorth_activity = []
        a_vals_hjorth_activity = []
        n_vals_hjorth_activity = []

        m_vals_hjorth_mobility = []
        a_vals_hjorth_mobility = []
        n_vals_hjorth_mobility = []

        m_vals_hjorth_complexity = []
        a_vals_hjorth_complexity = []
        n_vals_hjorth_complexity = []

        m_vals_DFA = []
        a_vals_DFA = []
        n_vals_DFA = []

        m_vals_PFD = []
        a_vals_PFD = []
        n_vals_PFD = []

        m_vals_LZC = []
        a_vals_LZC = []
        n_vals_LZC = []

        m_vals_HFD = []
        a_vals_HFD = []
        n_vals_HFD = []

        m_vals_sample_entropy = []
        a_vals_sample_entropy = []
        n_vals_sample_entropy = []

        m_vals_fuzzy_entropy = []
        a_vals_fuzzy_entropy = []
        n_vals_fuzzy_entropy = []

        m_vals_spectral_entropy = []
        a_vals_spectral_entropy = []
        n_vals_spectral_entropy = []

        m_vals_wave_entropy = []
        a_vals_wave_entropy = []
        n_vals_wave_entropy = []

        for k in range(len(all_files)):
            # analyse awake data
            for i in range(len(all_files[k])):
                temp_df = pd.read_pickle(all_files[k][i], compression='bz2')

                # delta
                val = self._eeg_mean_aux(temp_df, "bandPower(){}_0_delta", "bandPower(){}_1_delta", "bandPower(){}_2_delta")
                if k == 0:
                    m_vals_delta.append(val)
                elif k ==1:
                    a_vals_delta.append(val)
                elif k ==2:
                    n_vals_delta.append(val)
                
                # theta
                val = self._eeg_mean_aux(temp_df, "bandPower(){}_0_theta", "bandPower(){}_1_theta", "bandPower(){}_2_theta")
                if k == 0:
                    m_vals_theta.append(val)
                elif k ==1:
                    a_vals_theta.append(val)
                elif k ==2:
                    n_vals_theta.append(val)

                # alpha
                val = self._eeg_mean_aux(temp_df, "bandPower(){}_0_alpha", "bandPower(){}_1_alpha", "bandPower(){}_2_alpha")
                if k == 0:
                    m_vals_alpha.append(val)
                elif k ==1:
                    a_vals_alpha.append(val)
                elif k ==2:
                    n_vals_alpha.append(val)
                
                # beta
                val = self._eeg_mean_aux(temp_df, "bandPower(){}_0_beta", "bandPower(){}_1_beta", "bandPower(){}_2_beta")
                if k == 0:
                    m_vals_beta.append(val)
                elif k ==1:
                    a_vals_beta.append(val)
                elif k ==2:
                    n_vals_beta.append(val)

                #hjorth Activity
                val = self._eeg_mean_aux(temp_df, "hjorthActivity(){}_0", "hjorthActivity(){}_1", "hjorthActivity(){}_2")
                if k == 0:
                    m_vals_hjorth_activity.append(val)
                elif k ==1:
                    a_vals_hjorth_activity.append(val)
                elif k ==2:
                    n_vals_hjorth_activity.append(val)
                
                #hjorth Mobility
                val = self._eeg_mean_aux(temp_df, "hjorthMobility(){}_0", "hjorthMobility(){}_1", "hjorthMobility(){}_2")
                if k == 0:
                    m_vals_hjorth_mobility.append(val)
                elif k ==1:
                    a_vals_hjorth_mobility.append(val)
                elif k ==2:
                    n_vals_hjorth_mobility.append(val)
                
                #hjorth Complexity
                val = self._eeg_mean_aux(temp_df, "hjorthComplexity(){}_0", "hjorthComplexity(){}_1", "hjorthComplexity(){}_2")
                if k == 0:
                    m_vals_hjorth_complexity.append(val)
                elif k ==1:
                    a_vals_hjorth_complexity.append(val)
                elif k ==2:
                    n_vals_hjorth_complexity.append(val)
                
                # DFA
                val = self._eeg_mean_aux(temp_df, "DFA(){}_0", "DFA(){}_1", "DFA(){}_2")
                if k == 0:
                    m_vals_DFA.append(val)
                elif k ==1:
                    a_vals_DFA.append(val)
                elif k ==2:
                    n_vals_DFA.append(val)
                
                #PFD
                val = self._eeg_mean_aux(temp_df, "PFD(){}_0", "PFD(){}_1", "PFD(){}_2")
                if k == 0:
                    m_vals_PFD.append(val)
                elif k ==1:
                    a_vals_PFD.append(val)
                elif k ==2:
                    n_vals_PFD.append(val)

                #LZC
                val = self._eeg_mean_aux(temp_df, "LZC(){}_0", "LZC(){}_1", "LZC(){}_2")
                if k == 0:
                    m_vals_LZC.append(val)
                elif k ==1:
                    a_vals_LZC.append(val)
                elif k ==2:
                    n_vals_LZC.append(val)
                
                #HFD
                val = self._eeg_mean_aux(temp_df, "HFD(){}_0", "HFD(){}_1", "HFD(){}_2")
                if k == 0:
                    m_vals_HFD.append(val)
                elif k ==1:
                    a_vals_HFD.append(val)
                elif k ==2:
                    n_vals_HFD.append(val)
                
                #sample_entropy
                val = self._eeg_mean_aux(temp_df, "sampEn(){}_0", "sampEn(){}_1", "sampEn(){}_2")
                if k == 0:
                    m_vals_sample_entropy.append(val)
                elif k ==1:
                    a_vals_sample_entropy.append(val)
                elif k ==2:
                    n_vals_sample_entropy.append(val)
                
                #fuzzy entropy
                val = self._eeg_mean_aux(temp_df, "fuzzy0", "fuzzy1", "fuzzy2")
                if k == 0:
                    m_vals_fuzzy_entropy.append(val)
                elif k ==1:
                    a_vals_fuzzy_entropy.append(val)
                elif k ==2:
                    n_vals_fuzzy_entropy.append(val)
                
                #spectral entropy
                val = self._eeg_mean_aux(temp_df, "Sp_ent0", "Sp_ent1", "Sp_ent2")
                if k == 0:
                    m_vals_spectral_entropy.append(val)
                elif k ==1:
                    a_vals_spectral_entropy.append(val)
                elif k ==2:
                    n_vals_spectral_entropy.append(val)

                #wave entropy
                val = self._eeg_mean_aux(temp_df, "wave_ent0", "wave_ent1", "wave_ent2")
                if k == 0:
                    m_vals_wave_entropy.append(val)
                elif k ==1:
                    a_vals_wave_entropy.append(val)
                elif k ==2:
                    n_vals_wave_entropy.append(val)




        m_col = pd.DataFrame({"avg_delta_morning": m_vals_delta})
        a_col = pd.DataFrame({"avg_delta_afternoon": a_vals_delta})
        n_col = pd.DataFrame({"avg_delta_night": n_vals_delta})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_theta_morning": m_vals_theta})
        a_col = pd.DataFrame({"avg_theta_afternoon": a_vals_theta})
        n_col = pd.DataFrame({"avg_theta_night": n_vals_theta})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_alpha_morning": m_vals_alpha})
        a_col = pd.DataFrame({"avg_alpha_afternoon": a_vals_alpha})
        n_col = pd.DataFrame({"avg_alpha_night": n_vals_alpha})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_beta_morning": m_vals_beta})
        a_col = pd.DataFrame({"avg_beta_afternoon": a_vals_beta})
        n_col = pd.DataFrame({"avg_beta_night": n_vals_beta})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_hjorth_activity_morning": m_vals_hjorth_activity})
        a_col = pd.DataFrame({"avg_hjorth_activity_afternoon": a_vals_hjorth_activity})
        n_col = pd.DataFrame({"avg_hjorth_activity_night": n_vals_hjorth_activity})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_hjorth_mobility_morning": m_vals_hjorth_mobility})
        a_col = pd.DataFrame({"avg_hjorth_mobility_afternoon": a_vals_hjorth_mobility})
        n_col = pd.DataFrame({"avg_hjorth_mobility_night": n_vals_hjorth_mobility})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_hjorth_complexity_morning": m_vals_hjorth_complexity})
        a_col = pd.DataFrame({"avg_hjorth_complexity_afternoon": a_vals_hjorth_complexity})
        n_col = pd.DataFrame({"avg_hjorth_complexity_night": n_vals_hjorth_complexity})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_DFA_morning": m_vals_DFA})
        a_col = pd.DataFrame({"avg_DFA_afternoon": a_vals_DFA})
        n_col = pd.DataFrame({"avg_DFA_night": n_vals_DFA})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_PFD_morning": m_vals_PFD})
        a_col = pd.DataFrame({"avg_PFD_afternoon": a_vals_PFD})
        n_col = pd.DataFrame({"avg_PFD_night": n_vals_PFD})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_LZC_morning": m_vals_LZC})
        a_col = pd.DataFrame({"avg_LZC_afternoon": a_vals_LZC})
        n_col = pd.DataFrame({"avg_LZC_night": n_vals_LZC})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_HFD_morning": m_vals_HFD})
        a_col = pd.DataFrame({"avg_HFD_afternoon": a_vals_HFD})
        n_col = pd.DataFrame({"avg_HFD_night": n_vals_HFD})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_sample_entropy_morning": m_vals_sample_entropy})
        a_col = pd.DataFrame({"avg_sample_entropy_afternoon": a_vals_sample_entropy})
        n_col = pd.DataFrame({"avg_sample_entropy_night": n_vals_sample_entropy})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_fuzzy_entropy_morning": m_vals_fuzzy_entropy})
        a_col = pd.DataFrame({"avg_fuzzy_entropy_afternoon": a_vals_fuzzy_entropy})
        n_col = pd.DataFrame({"avg_fuzzy_entropy_night": n_vals_fuzzy_entropy})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_spectral_entropy_morning": m_vals_spectral_entropy})
        a_col = pd.DataFrame({"avg_spectral_entropy_afternoon": a_vals_spectral_entropy})
        n_col = pd.DataFrame({"avg_spectral_entropy_night": n_vals_spectral_entropy})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)

        m_col = pd.DataFrame({"avg_wave_entropy_morning": m_vals_wave_entropy})
        a_col = pd.DataFrame({"avg_wave_entropy_afternoon": a_vals_wave_entropy})
        n_col = pd.DataFrame({"avg_wave_entropy_night": n_vals_wave_entropy})
        self.df = pd.concat([self.df, m_col, a_col, n_col], axis=1)



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