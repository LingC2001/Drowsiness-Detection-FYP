from t_test import remove_outlier
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Reading and creating dataframe for the data
    df_main = pd.read_excel("kss_vs_time_pre.xlsx")
    df_out = pd.DataFrame()
    row_headers = ["n_sample time 1", "n_sample time 2", "mean_time_1", "std_time_1", "mean_time_2", "std_time_2","t-val", "p-val","alternative", "usable?"]
    row_col = pd.DataFrame({"time_of_day":row_headers})
    df_out = pd.concat([df_out, row_col], axis = 1)
    df_main.head(1)
    # Extracting and cleaning out the data
    morning_header = df_main.columns[0]
    afternoon_header = df_main.columns[1]
    night_header = df_main.columns[2]

    cleaned_morning_data = (df_main[morning_header].to_numpy()).tolist()
    cleaned_afternoon_data = (df_main[afternoon_header].to_numpy()).tolist()
    cleaned_night_data = (df_main[night_header].to_numpy()).tolist()

    # Plotting out the data
    morning_col = pd.DataFrame({morning_header:cleaned_morning_data})
    afternoon_col = pd.DataFrame({afternoon_header:cleaned_afternoon_data})
    night_col = pd.DataFrame({night_header:cleaned_night_data})
    df = pd.concat([morning_col, afternoon_col, night_col], axis=1)

    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=[morning_header, afternoon_header, night_header])
    df_melt.columns = ['index', 'Time of Day', "KSS val"]

    ax = sns.boxplot(x="Time of Day", y="KSS val", data=df_melt, color='#e3d7e5')
    ax = sns.swarmplot(x="Time of Day", y="KSS val",  data=df_melt)
    title_string = "KSS val" + " vs Time of Day for pre-experiment"
    ax.set_title(title_string) 
    plt.show()   

    cleaned_morning_data_nan = np.array(cleaned_morning_data)
    cleaned_morning_data = []
    for i in range(cleaned_morning_data_nan.shape[0]):
        if not np.isnan(cleaned_morning_data_nan[i]):
            cleaned_morning_data.append(cleaned_morning_data_nan[i])  

    cleaned_afternoon_data_nan = np.array(cleaned_afternoon_data)
    cleaned_afternoon_data = []
    for i in range(cleaned_afternoon_data_nan.shape[0]):
        if not np.isnan(cleaned_afternoon_data_nan[i]):
            cleaned_afternoon_data.append(cleaned_afternoon_data_nan[i])

    cleaned_night_data_nan = np.array(cleaned_night_data)
    cleaned_night_data = []
    for i in range(cleaned_night_data_nan.shape[0]):
        if not np.isnan(cleaned_night_data_nan[i]):
            cleaned_night_data.append(cleaned_night_data_nan[i])

    mean_morning = np.mean(cleaned_morning_data)
    std_morning = np.std(cleaned_morning_data)
    mean_afternoon = np.mean(cleaned_afternoon_data)
    std_afternoon = np.std(cleaned_afternoon_data)
    mean_night = np.mean(cleaned_night_data)
    std_night = np.std(cleaned_night_data)

    # Conducting t_test for every pair possible

    ## morning vs afternoon
    t_val_less_ma, p_val_less_ma = scipy.stats.ttest_ind(np.array(cleaned_morning_data), np.array(cleaned_afternoon_data), axis=0, equal_var=True, nan_policy='omit', alternative='less', permutations=None, random_state=None, trim=0)
    t_val_more_ma, p_val_more_ma = scipy.stats.ttest_ind(np.array(cleaned_morning_data), np.array(cleaned_afternoon_data), axis=0, equal_var=True, nan_policy='omit', alternative='greater', permutations=None, random_state=None, trim=0)
    t_val_two_ma, p_val_two_ma = scipy.stats.ttest_ind(np.array(cleaned_morning_data), np.array(cleaned_afternoon_data), axis=0, equal_var=True, nan_policy='omit', permutations=None, random_state=None, trim=0)

    p_val_ma = np.min([p_val_less_ma, p_val_more_ma, p_val_two_ma])
    id_p_ma = np.argmin([p_val_less_ma, p_val_more_ma, p_val_two_ma])
    t_vec_ma = [t_val_less_ma, t_val_more_ma, t_val_two_ma]
    t_val_ma = t_vec_ma[id_p_ma]

    if id_p_ma==0:
        print_alternative = "less"
    elif id_p_ma==1:
        print_alternative = "greater"
    else:
        print_alternative = "two sided"

    flagger_ma = False
    if p_val_ma<0.05:
        flagger_ma = True

    array_to_save_ma = [len(cleaned_morning_data), len(cleaned_afternoon_data), mean_morning, std_morning, mean_afternoon, std_afternoon, t_val_ma, p_val_ma, print_alternative, flagger_ma]
    new_col = pd.DataFrame({"morn vs afternoon": array_to_save_ma})
    df_out = pd.concat([df_out, new_col], axis=1)

    ## morning vs night
    t_val_less_mn, p_val_less_mn = scipy.stats.ttest_ind(np.array(cleaned_morning_data), np.array(cleaned_night_data), axis=0, equal_var=True, nan_policy='omit', alternative='less', permutations=None, random_state=None, trim=0)
    t_val_more_mn, p_val_more_mn = scipy.stats.ttest_ind(np.array(cleaned_morning_data), np.array(cleaned_night_data), axis=0, equal_var=True, nan_policy='omit', alternative='greater', permutations=None, random_state=None, trim=0)
    t_val_two_mn, p_val_two_mn = scipy.stats.ttest_ind(np.array(cleaned_morning_data), np.array(cleaned_night_data), axis=0, equal_var=True, nan_policy='omit', permutations=None, random_state=None, trim=0)

    p_val_mn = np.min([p_val_less_mn, p_val_more_mn, p_val_two_mn])
    id_p_mn = np.argmin([p_val_less_mn, p_val_more_mn, p_val_two_mn])
    t_vec_mn = [t_val_less_mn, t_val_more_mn, t_val_two_mn]
    t_val_mn = t_vec_mn[id_p_mn]

    if id_p_mn==0:
        print_alternative = "less"
    elif id_p_mn==1:
        print_alternative = "greater"
    else:
        print_alternative = "two sided"

    flagger_mn = False
    if p_val_mn<0.05:
        flagger_mn = True

    array_to_save_mn = [len(cleaned_morning_data), len(cleaned_night_data), mean_morning, std_morning, mean_night, std_night, t_val_mn, p_val_mn, print_alternative, flagger_mn]
    new_col = pd.DataFrame({"morn vs night": array_to_save_mn})
    df_out = pd.concat([df_out, new_col], axis=1)

    t_val_less_an, p_val_less_an = scipy.stats.ttest_ind(np.array(cleaned_afternoon_data), np.array(cleaned_night_data), axis=0, equal_var=True, nan_policy='omit', alternative='less', permutations=None, random_state=None, trim=0)
    t_val_more_an, p_val_more_an = scipy.stats.ttest_ind(np.array(cleaned_afternoon_data), np.array(cleaned_night_data), axis=0, equal_var=True, nan_policy='omit', alternative='greater', permutations=None, random_state=None, trim=0)
    t_val_two_an, p_val_two_an = scipy.stats.ttest_ind(np.array(cleaned_afternoon_data), np.array(cleaned_night_data), axis=0, equal_var=True, nan_policy='omit', permutations=None, random_state=None, trim=0)

    p_val_an = np.min([p_val_less_an, p_val_more_an, p_val_two_an])
    id_p_an = np.argmin([p_val_less_an, p_val_more_an, p_val_two_an])
    t_vec_an = [t_val_less_an, t_val_more_an, t_val_two_an]
    t_val_an = t_vec_an[id_p_an]

    if id_p_an==0:
        print_alternative = "less"
    elif id_p_an==1:
        print_alternative = "greater"
    else:
        print_alternative = "two sided"

    flagger_an = False
    if p_val_an<0.05:
        flagger_an = True

    array_to_save_an = [len(cleaned_afternoon_data), len(cleaned_night_data), mean_afternoon, std_afternoon, mean_night, std_night, t_val_an, p_val_an, print_alternative, flagger_an]
    new_col = pd.DataFrame({"afternoong vs night": array_to_save_an})
    df_out = pd.concat([df_out, new_col], axis=1)

    df_out.to_csv("t_test_kss_time_pre.csv")
