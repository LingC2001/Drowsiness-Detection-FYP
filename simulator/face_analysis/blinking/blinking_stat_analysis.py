import pandas as pd
import numpy as np
import math
import scipy

def remove_outlier(arr):
    original_len = arr.shape[0]
    new_arr = []
    for i in range(original_len):
        if not math.isnan(arr[i]):
            new_arr.append(arr[i])
    new_arr = np.array(new_arr)

    # # using IQR
    # p95, p5 = np.percentile(new_arr, [95, 5])
    # new_arr = new_arr[new_arr < p95]
    # new_arr = new_arr[new_arr > p5]

    # Using std
    arr_std = np.std(new_arr)
    arr_mean = np.mean(new_arr)
    new_arr = new_arr[new_arr > max(arr_mean - 2.5*arr_std, 2)]
    new_arr = new_arr[new_arr < arr_mean + 2.5*arr_std]

    while new_arr.shape[0] < original_len:
        new_arr = np.append(new_arr, np.nan)

    return new_arr

def analyse_time_of_day():
    # load data file
    df = pd.read_csv("blinking/M_A_N_vs_blinking_rate.csv")

    # Remove outliers
    new_col = remove_outlier(df["Morning"].to_numpy()).tolist()
    df['Morning'] = new_col
    new_col = remove_outlier(df["Afternoon"].to_numpy()).tolist()
    df['Afternoon'] = new_col
    new_col = remove_outlier(df["Night"].to_numpy()).tolist()
    df['Night'] = new_col


    # reshape the d dataframe suitable for statsmodels package 
    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['Morning', 'Afternoon', 'Night'])
    # replace column names
    df_melt.columns = ['index', 'Time of Day', 'Average Blinking Rate']

    # generate a boxplot to see the data distribution by treatments. Using boxplot, we can 
    # easily detect the differences between different treatments
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = sns.boxplot(x='Time of Day', y='Average Blinking Rate', data=df_melt, color='#99c2a2')
    ax = sns.swarmplot(x="Time of Day", y="Average Blinking Rate", data=df_melt, color='#7d0013')
    plt.show()
    print(df)

def analyse_KSS():
    # load data file
    df = pd.read_csv("blinking/KSS_vs_blinking_rate.csv")

    # Remove outliers
    new_col = remove_outlier(df["KSS<=6"].to_numpy()).tolist()
    df['KSS<=6'] = new_col
    new_col = remove_outlier(df["KSS>=7"].to_numpy()).tolist()
    df['KSS>=7'] = new_col


    # reshape the d dataframe suitable for statsmodels package 
    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['KSS<=6', 'KSS>=7'])
    # replace column names
    df_melt.columns = ['index', 'Drowsiness Level', 'Average Blinking Rate']

    # generate a boxplot to see the data distribution by treatments. Using boxplot, we can 
    # easily detect the differences between different treatments
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = sns.boxplot(x='Drowsiness Level', y='Average Blinking Rate', data=df_melt, color='#99c2a2')
    ax = sns.swarmplot(x="Drowsiness Level", y="Average Blinking Rate", data=df_melt, color='#7d0013')
    ax.set_title("Average Blinking Rate vs Drowsiness Level")
    plt.show()
    print(df)

    kss6_data = df['KSS<=6'].to_numpy()
    kss6_data_filtered = []
    for i in range(kss6_data.shape[0]):
        if not np.isnan(kss6_data[i]):
            kss6_data_filtered.append(kss6_data[i])

    kss7_data = df['KSS>=7'].to_numpy()
    kss7_data_filtered = []
    for i in range(kss7_data.shape[0]):
        if not np.isnan(kss7_data[i]):
            kss7_data_filtered.append(kss7_data[i])

    print("kss6 mean:" + str(np.mean(kss6_data_filtered)),"std: " + str(np.std(kss6_data_filtered)))
    print("kss7 mean:" + str(np.mean(kss7_data_filtered)),"std: " + str(np.std(kss7_data_filtered)))

    t_val, p_val = scipy.stats.ttest_ind(np.array(kss6_data_filtered), np.array(kss7_data_filtered), axis=0, equal_var=True, nan_policy='omit', permutations=None, random_state=None, alternative='less', trim=0)
    print("t-val:" + str(t_val),"p-val: " + str(p_val))


if __name__ == "__main__":
    # analyse_time_of_day()
    analyse_KSS()
# import scipy.stats as stats
# # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
# morning_vals = df['Morning'].to_numpy().tolist()
# new_morning_vals = []
# for i in range(len(morning_vals)):
#     if not math.isnan(morning_vals[i]):
#         new_morning_vals.append(morning_vals[i])
# morning_vals = np.array(new_morning_vals)

# morning_vals = df['Afternoon'].to_numpy().tolist()
# new_morning_vals = []
# for i in range(len(morning_vals)):
#     if not math.isnan(morning_vals[i]):
#         new_morning_vals.append(morning_vals[i])
# afternoon_values = np.array(new_morning_vals)

# morning_vals = df['Morning'].to_numpy().tolist()
# new_morning_vals = []
# for i in range(len(morning_vals)):
#     if not math.isnan(morning_vals[i]):
#         new_morning_vals.append(morning_vals[i])
# night_vals = np.array(new_morning_vals)

# fvalue, pvalue = stats.f_oneway(morning_vals, afternoon_values, night_vals)


# print(df['Morning'])
# print(fvalue, pvalue)

# # get ANOVA table as R like output
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

# # Ordinary Least Squares (OLS) model
# model = ols('value ~ Morning(Time of Day)', data=df_melt).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)