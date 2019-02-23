import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygama
#from waffle.processing import *
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
from scipy import stats
import os
import h5py
cal = [1173, 1332, 1460]
adc = [211, 246, 272]
cal = [1332, 1460, 2614]
adc = [257, 282, 503]
opt_tau = 1655.4
#44.8
def main():
    df = pd.read_hdf("test_data_dt.h5", key="data")

    slope, intercept, r_value, p_value, std_err = stats.linregress(adc, cal)
    df['trap_max'] = np.float64(df['trap_max']) * slope + intercept
    df['drift_time'] = df['drift_time'] * 10
    print(df.columns)

    cut =  (df['trap_max'] > 1420) & (df['trap_max'] < 1500)
    #cut = cut & (df['drift_time'] > 450) & (df['drift_time'] < 1500)
    df = df[cut]

    slope, intercept, r_value, p_value, std_err = stats.linregress(df['drift_time'], df['trap_max'])

    taus = np.linspace(-1000,2000, 1000)
    fwhms = []
    for tau in taus:
        fwhms.append(2.35*np.std(df['trap_max'] * np.exp(df['drift_time'] * (-tau / 10000000))))
    tau = taus[np.argmin(fwhms)]

    plt.figure(1)
    plt.scatter(taus, fwhms)
    print(taus[np.argmin(fwhms)])
    plt.figure(3)
    plt.hist(df['trap_max'] * np.exp(df['drift_time'] * (-33 / 10000000)), bins=40)
    plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def time(df):
    drift_time = []

    for wf in df['waveform']:
        dt = []
        for i in range(50,len(wf)):

            if wf[i] > (.9 * np.amax(wf)):
                dt.append(i)
                continue
        try:
            drift_time.append(dt[0])
        except:
            drift_time.append(-100)
    
    df['drift_time'] = drift_time - df['t0']


if __name__ == '__main__':
    main()
