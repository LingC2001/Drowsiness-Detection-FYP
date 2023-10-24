import pandas as pd
import numpy as np
import math
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

def remove_outlier(arr):
    original_len = np.array(arr).shape[0]
    new_arr = []
    for i in range(original_len):
        if not math.isnan(arr[i]):
            new_arr.append(arr[i])
    new_arr = np.array(new_arr)
    # print(new_arr)

    p95, p5 = np.percentile(new_arr, [95, 5])
    new_arr = new_arr[new_arr < p95]
    new_arr = new_arr[new_arr > p5]

    while new_arr.shape[0] < original_len:
        new_arr = np.append(new_arr, np.nan)
    return new_arr

## CODE TO RUN

if __name__ == "__main__":
    df_main = pd.read_csv("45s/data/features_vs_time_of_day_45.csv")
    df_out = pd.DataFrame()
    row_headers = ["n_sameple awake", "n_sameple drowsy", "mean_awake", "std_awake", "mean_drowsy", "std_drowsy","t-val", "p-val","alternative", "usable?"]
    row_col = pd.DataFrame({"feature_used":row_headers})
    df_out = pd.concat([df_out, row_col], axis = 1)
    counter = 0
    for i in range(0,len(df_main.columns),3):
        print(i)
        if i==384:
            continue
        # if i<120:
        #     continue
        # Obtaining the data
        awake_header = df_main.columns[i+1]
        drowsy_header = df_main.columns[i+2]

        cleaned_awake_data = remove_outlier(df_main[awake_header].to_numpy()).tolist()
        cleaned_drowsy_data = remove_outlier(df_main[drowsy_header].to_numpy()).tolist()


        # print(df)

         

        cleaned_awake_data_nan = np.array(cleaned_awake_data)
        cleaned_awake_data = []
        for i in range(cleaned_awake_data_nan.shape[0]):
            if not np.isnan(cleaned_awake_data_nan[i]):
                cleaned_awake_data.append(cleaned_awake_data_nan[i])

        cleaned_drowsy_data_nan = np.array(cleaned_drowsy_data)
        cleaned_drowsy_data = []
        for i in range(cleaned_drowsy_data_nan.shape[0]):
            if not np.isnan(cleaned_drowsy_data_nan[i]):
                cleaned_drowsy_data.append(cleaned_drowsy_data_nan[i])

        # Conducting t-tests
        t_val_less, p_val_less = scipy.stats.ttest_ind(np.array(cleaned_awake_data), np.array(cleaned_drowsy_data), axis=0, equal_var=True, nan_policy='omit', alternative='less', permutations=None, random_state=None, trim=0)
        t_val_more, p_val_more = scipy.stats.ttest_ind(np.array(cleaned_awake_data), np.array(cleaned_drowsy_data), axis=0, equal_var=True, nan_policy='omit', alternative='greater', permutations=None, random_state=None, trim=0)
        t_val_two, p_val_two = scipy.stats.ttest_ind(np.array(cleaned_awake_data), np.array(cleaned_drowsy_data), axis=0, equal_var=True, nan_policy='omit', permutations=None, random_state=None, trim=0)

        p_val = np.min([p_val_less, p_val_more, p_val_two])
        id_p = np.argmin([p_val_less, p_val_more, p_val_two])
        t_vec = [t_val_less, t_val_more, t_val_two]
        t_val = t_vec[id_p]

        if id_p==0:
            print_alternative = "less"
        elif id_p==1:
            print_alternative = "greater"
        else:
            print_alternative = "two sided"

        flagger = False
        if p_val<0.05:
            flagger = True
            counter +=1
        else:
            continue

        # awake_col = pd.DataFrame({awake_header:cleaned_awake_data})
        # drowsy_col = pd.DataFrame({drowsy_header:cleaned_drowsy_data})
        # df = pd.concat([awake_col, drowsy_col], axis=1)

        # # Reshape the dataframe suitable for statsmodels package 
        # df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=[awake_header, drowsy_header])
        # df_melt.columns = ['index', 'Drowsiness Level', awake_header[0:-6]]

        # # Showing the box plots
        # ax = sns.boxplot(x="Drowsiness Level", y=awake_header[0:-6], data=df_melt, color='#e3d7e5')
        # ax = sns.swarmplot(x="Drowsiness Level", y=awake_header[0:-6], data=df_melt)
        # title_string = "NNI for 50ms RRI difference" + " vs Drowsiness Level"
        # ax.set_title(title_string)   
        # plt.show()


        # Printing out values for visualising
        mean_awake = np.mean(cleaned_awake_data)
        std_awake = np.std(cleaned_awake_data)
        mean_drowsy = np.mean(cleaned_drowsy_data)
        std_drowsy = np.std(cleaned_drowsy_data)
        # print("For " + awake_header[0:-6] + " data,")
        # print("kss6 mean:" + str(mean_awake), "std: " + str(std_awake))
        # print("kss7 mean:" + str(mean_drowsy), "std: " + str(std_drowsy))
        # print("t-value: " + str(t_val), "p-value" + str(p_val))

        # plt.show()

        # Saving the values
        array_to_save = [len(cleaned_awake_data), len(cleaned_drowsy_data), mean_awake, std_awake, mean_drowsy, std_drowsy, t_val, p_val, print_alternative, flagger]
        new_col = pd.DataFrame({awake_header[0:-6]: array_to_save})
        df_out = pd.concat([df_out, new_col], axis=1)

        df = 0

    print(counter)
    # df_out.to_csv("45s/result/t_test_features_vs_kss_45.csv")
