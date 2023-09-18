import pandas as pd
import numpy as np
from toolz import interleave

features_labels_filename = "features_and_labels_15s.csv"

df = pd.read_csv(features_labels_filename)

print(df)

# features split by kss
awake_column_headers = [x+"_awake" for x in df.columns.tolist()]
awake_df = pd.DataFrame(columns=awake_column_headers)

drowsy_column_headers = [x+"_drowsy" for x in df.columns.tolist()]
drowsy_df = pd.DataFrame(columns=drowsy_column_headers)

for i in range(len(df)):
    row = df.iloc[i].tolist()
    if row[-1] >= 6:
        drowsy_df.loc[len(drowsy_df)] = row
    else:
        awake_df.loc[len(awake_df)] = row
kss_df = pd.concat([awake_df, drowsy_df], axis=1)[list(interleave([awake_df, drowsy_df]))]
kss_df.to_csv("features_vs_kss.csv", index=False) 


# features split by time of day
m_column_headers = [x+"_m" for x in df.columns.tolist()]
m_df = pd.DataFrame(columns=m_column_headers)

a_column_headers = [x+"_a" for x in df.columns.tolist()]
a_df = pd.DataFrame(columns=a_column_headers)

n_column_headers = [x+"_n" for x in df.columns.tolist()]
n_df = pd.DataFrame(columns=n_column_headers)

for i in range(len(df)):
    row = df.iloc[i].tolist()
    if row[-2] == "morning":
        m_df.loc[len(m_df)] = row
    elif row[-2] == "afternoon":
        a_df.loc[len(a_df)] = row
    elif row[-2] == "night":
        n_df.loc[len(n_df)] = row

tod_df = pd.concat([m_df, a_df, n_df], axis=1)[list(interleave([m_df, a_df, n_df]))]
tod_df.to_csv("features_vs_time_of_day.csv", index=False) 