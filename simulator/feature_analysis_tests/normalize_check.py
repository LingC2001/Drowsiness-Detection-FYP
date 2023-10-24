import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kstest
from scipy.stats import shapiro

def remove_outlier(arr):
    original_len = np.array(arr).shape[0]
    new_arr = []
    for i in range(original_len):
        if not math.isnan(arr[i]):
            new_arr.append(arr[i])
    new_arr = np.array(new_arr)

    p95, p5 = np.percentile(new_arr, [95, 5])
    new_arr = new_arr[new_arr < p95]
    new_arr = new_arr[new_arr > p5]

    while new_arr.shape[0] < original_len:
        new_arr = np.append(new_arr, np.nan)

    return new_arr

if __name__ == "__main__":
    df_main = pd.read_csv("features_and_labels_30s.csv")
    df_out = pd.DataFrame()
    row_headers = ["mean", "std", "stat_ks","p-val_ks", "stat_sw", "p_val_sw", "distributed_normally?"]
    row_col = pd.DataFrame({"feature_used":row_headers})
    df_out = pd.concat([df_out, row_col], axis = 1)
    counter = 0

    for i in range(0,len(df_main.columns)):
        if i==128:
            continue
        header = df_main.columns[i]
        print(i)
        cleaned_data = (df_main[header].to_numpy()).tolist()
        cleaned_data_nan = np.array(cleaned_data)
        cleaned_data = []
        for i in range(cleaned_data_nan.shape[0]):
            if not np.isnan(cleaned_data_nan[i]):
                cleaned_data.append(cleaned_data_nan[i])

        [statistic, p_val] = kstest(cleaned_data, 'norm')
        [stat_shap, p_val_shapiro] = shapiro(cleaned_data)

        flagger = False
        if p_val<=0.05:
            flagger = True
            counter += 1

        mean_data = np.mean(cleaned_data)
        std_data = np.std(cleaned_data)

        array_to_save = [mean_data, std_data, statistic, p_val, stat_shap, p_val_shapiro ,flagger]
        new_col = pd.DataFrame({header: array_to_save})
        df_out = pd.concat([df_out, new_col], axis=1)
    
    print(counter)
    print(len(df_main.columns))
    df_out.to_csv("results_30s/normal_checker.csv")