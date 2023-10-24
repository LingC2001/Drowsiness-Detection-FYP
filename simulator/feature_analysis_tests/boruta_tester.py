import pandas as pd
import numpy as np
import math
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

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
    df_main = pd.read_csv("features_and_labels_15s.csv")
    df_out = pd.DataFrame()

    for i in range(0,len(df_main.columns)):
        print(i)
        header = df_main.columns[i]

        trial_dataset = df_main.drop(header, axis=1)
        target = df_main[header]

        forest = RandomForestRegressor(n_jobs = -1,max_depth = len(df_main.columns))
        boruta = BorutaPy(estimator = forest, n_estimators = 'auto', max_iter = 20)

        print(np.array(target))
        boruta.fit(np.array(trial_dataset), np.array(target))

        green_area = trial_dataset.columns[boruta.support_].to_list()
        blue_area = trial_dataset.columns[boruta.support_weak_].to_list()

        array_to_save = [green_area, "", "", blue_area]
        new_col = pd.DataFrame({header: array_to_save})
        df_out = pd.concat([df_out, new_col], axis=1)
    
    df_out.to_csv("results_15s/boruta.csv")