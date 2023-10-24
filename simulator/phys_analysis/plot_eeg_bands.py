import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt

import scipy.signal as signal
df = pd.read_csv("DATA/bci_filtered/13m_filtered.csv", header=None)

eeg3 = df.iloc[50000:50500,4]

order = 8
fs = 125

#delta
low_pass_d = signal.butter(order, 4, 'lowpass', fs=fs, output='sos')
high_pass_d = signal.butter(order, 0.5, 'highpass', fs=fs, output='sos')

#theta
low_pass_t = signal.butter(order, 8, 'lowpass', fs=fs, output='sos')
high_pass_t = signal.butter(order, 4, 'highpass', fs=fs, output='sos')

#alpha
low_pass_a = signal.butter(order, 12, 'lowpass', fs=fs, output='sos')
high_pass_a = signal.butter(order, 8, 'highpass', fs=fs, output='sos')

#beta
low_pass_b = signal.butter(order, 35, 'lowpass', fs=fs, output='sos')
high_pass_b = signal.butter(order, 12, 'highpass', fs=fs, output='sos')

delta = signal.sosfilt(low_pass_d, eeg3)
delta = signal.sosfilt(high_pass_d, delta)

theta = signal.sosfilt(low_pass_t, eeg3)
theta = signal.sosfilt(high_pass_t, theta)

alpha = signal.sosfilt(low_pass_a, eeg3)
alpha = signal.sosfilt(high_pass_a, alpha)

beta = signal.sosfilt(low_pass_b, eeg3)
beta = signal.sosfilt(high_pass_b, beta)

fig, ax = plt.subplots(5,1)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.8)
ax[0].plot(eeg3)
ax[0].title.set_text('Participant 11 Raw EEG (4s)')
ax[0].xaxis.set_visible(False) # Hide only x axis
ax[1].plot(delta)
ax[1].title.set_text('Delta (0.5-4Hz)')
ax[1].xaxis.set_visible(False) # Hide only x axis
ax[2].plot(theta)
ax[2].title.set_text('Theta (4-8Hz)')
ax[2].xaxis.set_visible(False) # Hide only x axis
ax[3].plot(alpha)
ax[3].title.set_text('Alpha (8-12Hz)')
ax[3].xaxis.set_visible(False) # Hide only x axis
ax[4].plot(beta)
ax[4].title.set_text('Beta (12-35Hz)')
ax[4].xaxis.set_visible(False) # Hide only x axis
plt.show()