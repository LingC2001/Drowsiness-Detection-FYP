import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Reading and creating dataframe for the data
    df_main = pd.read_excel("kss_pre_post.xlsx")
    df_out = pd.DataFrame()
    row_headers = ["n_sample pre", "n_sample post", "mean_pre", "std_pre", "mean_post", "std_post","t-val", "p-val","alternative", "usable?"]
    row_col = pd.DataFrame({"Survey time":row_headers})
    df_out = pd.concat([df_out, row_col], axis = 1)
    df_main.head(1)
    # Extracting and cleaning out the data
    pre_header = df_main.columns[0]
    post_header = df_main.columns[1]

    cleaned_pre_data = (df_main[pre_header].to_numpy()).tolist()
    cleaned_post_data = (df_main[post_header].to_numpy()).tolist()

    pre_column = pd.DataFrame({pre_header:cleaned_pre_data})
    post_column = pd.DataFrame({post_header:cleaned_post_data})

    df = pd.concat([pre_column, post_column], axis=1)
    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=[pre_header, post_header])
    df_melt.columns = ['index', 'Time of Record', "KSS val"]

    ax = sns.boxplot(x="Time of Record", y="KSS val", data=df_melt, color='#99c2a2')
    ax = sns.swarmplot(x="Time of Record", y="KSS val",  data=df_melt, color='#7d0013')
    title_string = "KSS val" + " vs Time of Record for post-experiment"
    ax.set_title(title_string) 
    plt.show() 

    cleaned_pre_data_nan = np.array(cleaned_pre_data)
    cleaned_pre_data = []
    for i in range(cleaned_pre_data_nan.shape[0]):
        if not np.isnan(cleaned_pre_data_nan[i]):
            cleaned_pre_data.append(cleaned_pre_data_nan[i])  

    cleaned_post_data_nan = np.array(cleaned_post_data)
    cleaned_post_data = []
    for i in range(cleaned_post_data_nan.shape[0]):
        if not np.isnan(cleaned_post_data_nan[i]):
            cleaned_post_data.append(cleaned_post_data_nan[i])     

    mean_pre = np.mean(cleaned_pre_data)
    std_pre = np.std(cleaned_pre_data)
    mean_post = np.mean(cleaned_post_data)
    std_post = np.std(cleaned_post_data)

    t_val_less_ma, p_val_less_ma = scipy.stats.ttest_ind(np.array(cleaned_pre_data), np.array(cleaned_post_data), axis=0, equal_var=True, nan_policy='omit', alternative='less', permutations=None, random_state=None, trim=0)
    t_val_more_ma, p_val_more_ma = scipy.stats.ttest_ind(np.array(cleaned_pre_data), np.array(cleaned_post_data), axis=0, equal_var=True, nan_policy='omit', alternative='greater', permutations=None, random_state=None, trim=0)
    t_val_two_ma, p_val_two_ma = scipy.stats.ttest_ind(np.array(cleaned_pre_data), np.array(cleaned_post_data), axis=0, equal_var=True, nan_policy='omit', permutations=None, random_state=None, trim=0)

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

    array_to_save_ma = [len(cleaned_pre_data), len(cleaned_post_data), mean_pre, std_pre, mean_post, std_post, t_val_ma, p_val_ma, print_alternative, flagger_ma]
    new_col = pd.DataFrame({"pre vs post": array_to_save_ma})
    df_out = pd.concat([df_out, new_col], axis=1)

    df_out.to_csv("t_test_kss_pre_post.csv")
