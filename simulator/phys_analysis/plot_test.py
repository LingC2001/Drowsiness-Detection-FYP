import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("DATA/bci_filtered/8n_filtered.csv", header=None)
ecg = df.iloc[:, 0].to_numpy()
eeg = df.iloc[:, 4].to_numpy()
print(ecg)

plt.plot(ecg)
plt.show()
