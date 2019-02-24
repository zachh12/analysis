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

cal = [727.330, 860.564, 2614.553]
adc_ft = [1811, 2131, 6438]
adc_max = [1810, 2126, 6450]
#opt_tau = 1655.4

def main():
    df = pd.read_hdf("mjd_energy.h5")

    #plots(df)

    slope, intercept, r_value, p_value, std_err = stats.linregress(adc_max, cal)
    df['trap_max'] = df['trap_max'] * slope + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(adc_ft, cal)
    df['trap_ft'] = df['trap_ft'] * slope + intercept

    cut =  (df['trap_max'] > 2000)
    df = df[cut]
    #plt.hist(df['trap_max'], bins=60)
    #plt.show()
    print(np.std(df['trap_max']) * 2.35)
    print(np.std(df['trap_ft']) * 2.35)
    taus = np.linspace(-500,500, 1000)
    fwhms = []
    fwhms2 = []
    for tau in taus:
        fwhms.append(2.35*np.std(df['trap_max'] * np.exp(df['drift_time'] * (-tau / 10000000))))
        fwhms2.append(2.35*np.std(df['trap_ft'] * np.exp(df['drift_time'] * (-tau / 10000000))))
    tau = taus[np.argmin(fwhms)]
    print(np.min(fwhms), np.min(fwhms2))
    df['trap_max'] = df['trap_max'] * np.exp(df['drift_time'] * (-taus[np.argmin(fwhms)] / 10000000))
    df['trap_ft'] = df['trap_ft'] * np.exp(df['drift_time'] * (-taus[np.argmin(fwhms)] / 10000000))
    fwhms = []
    fwhms2 = []
    taus = np.linspace(-10000,20000, 1000)
    length = np.sqrt(np.square(df['hole_drift_length']))
    for tau in taus:
        fwhms.append(2.35*np.std(df['trap_max'] * np.exp(length * (-tau / 10000000))))
        fwhms2.append(2.35*np.std(df['trap_ft'] * np.exp(length * (-tau / 10000000))))
    tau = taus[np.argmin(fwhms)]
    print(np.min(fwhms), np.min(fwhms2))

def plots(df):
    plt.hist(df['trap_ft'], bins=400)
    plt.show()
    plt.hist(df['trap_max'], bins=400)
    plt.show()
    exit()

if __name__ == '__main__':
    main()
