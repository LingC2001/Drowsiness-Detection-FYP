import numpy as np
import matplotlib.pyplot as plt
import scipy
import pywt
import antropy as ent # https://github.com/raphaelvallat/antropy/blob/master/antropy/entropy.py
from scipy.signal import butter, lfilter,freqz
from wfdb import processing
#from pyhrv.hrv import hrv
from pywt import cwt
from ecgdetectors import Detectors
import hrv
from hrv.filters import moving_median
from hrv.classical import time_domain
import heartpy as hp
from scipy import signal
import mne
from mne.viz import plot_raw



def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def featureExtraction(rri, ECG, fs):
    time_domain = hrvanalysis.get_time_domain_features(rri)
    frequency_domain = hrvanalysis.get_frequency_domain_features(rri)
    other = hrvanalysis.get_csi_cvi_features(rri)

    HRV = [time_domain['mean_nni'], time_domain['cvnni']]
    Time_Features = [time_domain['cvsd'], time_domain['max_hr'], time_domain['mean_hr'], time_domain['median_nni'], time_domain['min_hr'], time_domain['nni_20'], time_domain['nni_50'], time_domain['pnni_20'],
                                                    time_domain['pnni_50'], time_domain['range_nni'], time_domain['rmssd'], time_domain['sdnn'], time_domain['sdsd'], time_domain['std_hr']]

    Frequency_Features = [frequency_domain['lf'], frequency_domain['hf'], frequency_domain['lf_hf_ratio'], frequency_domain['lfnu'], frequency_domain['hfnu'], frequency_domain['total_power'], frequency_domain['vlf']]
    Other_Features = [other['csi'], other['cvi']]

    # Retrieves a pre-defined feature configuration file to extract all available features
    cfg = tsfel.get_features_by_domain()

    # Extract features
    ECG = pd.DataFrame(ECG)
    X = tsfel.time_series_features_extractor(cfg, ECG, fs=fs)
    return np.float64(np.concatenate((np.array(HRV), np.array(Time_Features), np.array(Frequency_Features), np.array(Other_Features), np.array(X).flatten())))


def filter_signal(x, fs, fmin, fmax):
    fmin = float(fmin)
    fmax = float(fmax)
    fs = float(fs)
    b, a = signal.butter(2, [2 * fmin / fs, 2 * fmax / fs], 'bandpass')
    return signal.filtfilt(b, a, x)


def ECG_highpass(record, sfreq, nsample):
    order = 3
    lowcut = 0.1
    # highcut = 50
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    # high = highcut / nyq
    numerator_coeffs, denominator_coeffs = signal.butter(order, low, btype='high')

    sig = record
    filtered_record = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    return filtered_record


# Low pass butterworth function
def ECG_lowpass(record, sfreq, nsample):
    cutoff_freq = 30
    order = 10
    normalized_cutoff_freq = 2 * cutoff_freq / sfreq

    numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)

    time = np.linspace(0, nsample / sfreq, nsample, endpoint=False)

    sig = record
    filtered_record = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    return filtered_record


# Low pass anti-aliasing butterworth function
def ECG_lowpass_anti_aliasing(record, sfreq, nsample):
    cutoff_freq = 50
    order = 16
    normalized_cutoff_freq = 2 * cutoff_freq / sfreq

    numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)

    time = np.linspace(0, nsample / sfreq, nsample, endpoint=False)

    sig = record
    filtered_record= signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    return filtered_record


####################### FEATURE DEFINITIONS ###################################
"""TIME DOMAIN"""
#independent function to calculate RMSSD
def calc_rmssd(list):
    diff_nni = np.diff(list)#successive differences
    return np.sqrt(np.mean(diff_nni ** 2))
    
    
 #independent function to calculate AVRR   
def calc_avrr(list):
    return sum(list)/len(list)

 #independent function to calculate SDRR   
def calc_sdrr(list):
    return statistics.stdev(list)

 #independent function to calculate SKEW   
def calc_skew(list):
    return skew(list)

 #independent function to calculate KURT   
def calc_kurt(list):
    return kurtosis(list)

def calc_NNx(list):
    diff_nni = np.diff(list)
    return sum(np.abs(diff_nni) > 50)
    
def calc_pNNx(list):
    length_int = len(list)
    diff_nni = np.diff(list)
    nni_50 = sum(np.abs(diff_nni) > 50)
    return 100 * nni_50 / length_int

"""NON LINEAR DOMAIN"""
 #independent function to calculate SD1
def calc_SD1(list):
    diff_nn_intervals = np.diff(list)
    return np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
 #independent function to calculate SD2
def calc_SD2(list):
    diff_nn_intervals = np.diff(list)
    return np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                   diff_nn_intervals, ddof=1) ** 2)
    
 #independent function to calculate SD1/SD2
def calc_SD1overSD2(list):
      diff_nn_intervals = np.diff(list)
      sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
      sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                    diff_nn_intervals, ddof=1) ** 2)
      ratio_sd2_sd1 = sd2 / sd1
      return ratio_sd2_sd1
    
    
 #independent function to calculate CSI
def calc_CSI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                  diff_nn_intervals, ddof=1) ** 2)
    L=4 * sd1
    T=4 * sd2
    return L/T
       
 #independent function to calculate CVI
def calc_CVI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                  diff_nn_intervals, ddof=1) ** 2)
    L=4 * sd1
    T=4 * sd2
    return np.log10(L * T)
 
 #independent function to calculate modified CVI
def calc_modifiedCVI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                  diff_nn_intervals, ddof=1) ** 2)
    L=4 * sd1
    T=4 * sd2
    return L ** 2 / T

 
#sliding window function
def slidingWindow(sequence,winSize,step):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
 
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence\
                        length.")
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
    # Do the work
    for i in range(0,int(numOfChunks)*step,step):
        yield sequence[i:i+winSize]
        
####################### FEATURE EXTRACTION ####################################

def feature_extract(list_rri, winSize,step,feature):
    chunklist=list(slidingWindow(list_rri,winSize,step))
    featureList=[]
    if(feature=="RMSSD"):
        for sublist in chunklist:
            featureList.append(calc_rmssd(sublist))
    elif(feature=="AVRR"):
        for sublist in chunklist:
            featureList.append(calc_avrr(sublist))
    elif(feature=="SDRR"):
        for sublist in chunklist:
            featureList.append(calc_sdrr(sublist))
    elif(feature=="SKEW"):
        for sublist in chunklist:
            featureList.append(calc_skew(sublist))
    elif(feature=="KURT"):
        for sublist in chunklist:
            featureList.append(calc_kurt(sublist))
    elif(feature=="NNx"):
        for sublist in chunklist:
            featureList.append(calc_NNx(sublist))
    elif(feature=="pNNx"):
        for sublist in chunklist:
            featureList.append(calc_pNNx(sublist))
    elif(feature=="SD1"):
        for sublist in chunklist:
            featureList.append(calc_SD1(sublist))
    elif(feature=="SD2"):
        for sublist in chunklist:
            featureList.append(calc_SD2(sublist))
    elif(feature=="SD1/SD2"):
        for sublist in chunklist:
            featureList.append(calc_SD1overSD2(sublist))
    elif(feature=="CSI"):
        for sublist in chunklist:
            featureList.append(calc_CSI(sublist))
    elif(feature=="CVI"):
        for sublist in chunklist:
            featureList.append(calc_CVI(sublist))
    elif(feature=="modifiedCVI"):
        for sublist in chunklist:
            featureList.append(calc_modifiedCVI(sublist))
    return featureList    
  
########################### PLOTTING ##########################################
def plot_features(featureList,label):
    plt.title(label)
    plt.plot(featureList)
    plt.show()

###################### CALLING FEATURE METHODS ################################
def browsethroughSeizures(list_rri,winSize,step):
    features=["RMSSD","AVRR","SDRR","SKEW","KURT","NNx","pNNx","SD1","SD2",\
              "SD1/SD2","CSI","CVI","modifiedCVI"]
    for item in features:
        featureList=feature_extract(list_rri,winSize,step,item)
        plot_features(featureList,item)
#################### BAYESIAN CHANGE POINT DETECTION ##########################
####inspired by https://github.com/hildensia/bayesian_changepoint_detection
def bayesianOnFeatures(list_rri,winSize,step):
    features=["RMSSD","AVRR","SDRR","SKEW","KURT","NNx","pNNx","SD1","SD2",\
              "SD1/SD2","CSI","CVI","modifiedCVI"]
    for item in features:
        featureList=feature_extract(list_rri,winSize,step,item)
        featureList=np.asanyarray(featureList)
        Q, P, Pcp = ocpd.offline_changepoint_detection\
        (featureList, partial(ocpd.const_prior,l=(len(featureList)+1))\
         ,ocpd.gaussian_obs_log_likelihood, truncate=-40)
        fig, ax = plt.subplots(figsize=[15, 7])
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title(item)
        ax.plot(featureList[:])
        ax = fig.add_subplot(2, 1, 2, sharex=ax)
        ax.plot(np.exp(Pcp).sum(0))
        
#################### CHANGE POINT DETECTION ##########################

# sub = 1
# for c in range(sub):
#     # Collect the data for a subject
#     '''data, fs, s, stage = get_data(c+1, training)
#     raw1 = data.values
#     out = np.transpose(raw1)'''

#     raw_s = mne.io.read_raw_edf('1a_cleaned.edf')
#     fs = 250
    
#     data, times = raw_s[:]
#     data_use = -(data[0, :])
#     lener = np.floor(len(data_use)/fs)
#     data_use = data_use[0:250*int(lener)]

#     print(np.shape(data_use))


#     M11 = ECG_lowpass_anti_aliasing(data_use, fs, len(data_use))
#     M12, time = processing.resample_sig(data_use, fs, 200)
#     fs = 200
#     M1 = ECG_lowpass(M12, fs, len(data_use))
#     ECG = ECG_highpass(M1, fs, len(data_use))

