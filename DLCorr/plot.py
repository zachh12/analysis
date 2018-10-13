import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4.classic as dn
from scipy import stats
from scipy.stats import norm
import matplotlib.mlab as mlab

bad = [26384.0, 24579.0]
chan = 626
tau = 691.2138
taut = 187.443748

def main():
    plots()

def plots():

    plt.figure(1)
    df = getDataFrame()
    cut = df['avse'] < 3
    df = df[cut]
    #tune(df)
    ecorr, ecorrt = corr(df)
    FWHM(df, ecorr, ecorrt)

def tune(df):
    check = np.linspace(0, 1500, 50000)
    rez, rez2 = [], []
    for var in check:
        rez.append(np.std(df['ecal'] * np.exp(df['drift_length']*(var/10000000)))*2.35)
    for var in check:
        rez2.append(np.std(df['ecal'] * np.exp(df['drift_time'] * (var/10000000))) * 2.35)

    tau = check[np.argmin(rez)]
    tau2 = check[np.argmin(rez2)]
    #print(tau, tau2)
    print("Length: ", np.min(rez))
    print("Time: ", np.min(rez2))

def corr(df):
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['drift_length'], df['ecal'])
    print("Slope: ", slope)
    x = np.linspace(10, 50)
    y = x * slope + intercept
    regression = plt.plot(x, y, label="R: " + str(('%.5f'%(r_value))))
    plt.legend(loc='upper right')
    plt.scatter(df['drift_length'], df['ecal'])
    plt.xlim(20,)
    plt.xlabel("Drift Length (mm)")
    plt.ylabel("Energy (arb)")
    plt.title("Drift Length vs Energy @2614 keV Peak")
    plt.figure(2)
    ecorr = df['ecal'] * np.exp(df['drift_length']*(tau/10000000))
    ecorrt = df['ecal'] * np.exp(df['drift_time'] * (2.32213/100000))
    plt.scatter(df['drift_length'], ecorr)
    plt.title("Drift Length vs Length Corrected Energy")
    plt.xlabel("Drift Length (mm)")
    plt.ylabel("Energy (arb)")
    print("None: ", np.std(df['ecal'])*2.35)
    print("Time: ", np.std(ecorrt)*2.35)
    print("Length: ", np.std(ecorr)*2.35)
    #plt.figure(100)
    #plt.hist(df['ecal'] - np.mean(df['ecal']), alpha=.6, bins=20)
    #plt.hist(ecorrt - np.mean(ecorrt), alpha=.5, bins=20)
    #plt.hist(ecorr - np.mean(ecorr), alpha=.5, bins=12)
    plt.show()
    return ecorr, ecorrt

def FWHM(df, ecorr, ecorrt):
    energy = df.ecal - np.mean(df.ecal)
    ecorr = ecorr - np.mean(ecorr)
    ecorrt = ecorrt - np.mean(ecorrt)

    (mu, sigma) = norm.fit(ecorr)
    n, bins, patches = plt.hist(ecorr, 15, normed=1, facecolor='red', alpha=0)
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r', linewidth=2, label='Length-Based Correction, FWHM: 3.915 keV')

    (mu, sigma) = norm.fit(ecorrt)
    n, bins, patches = plt.hist(ecorr, 15, normed=1, facecolor='yellow', alpha=0)
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'g', linewidth=2, label='Time-Based Correction, FWHM: 3.541 keV')

    (mu, sigma) = norm.fit(energy)
    n, bins, patches = plt.hist(ecorr, 15, normed=1, facecolor='green', alpha=0)
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'b', linewidth=2, label='No Correction, FWHM: 3.153 keV')
    plt.legend(shadow=True, fancybox=True, loc='upper right', prop={'size': 7.5},)
    plt.title("Energy Resolution @2614 keV")
    plt.xlabel("Energy Centered at Mean")
    print("Resolution: ", sigma*2.35)
    plt.show()
def getDataFrame():
    name = "data/chan" + str(chan) + "data.h5"
    try:
        df = pd.read_hdf(name, key='data')
    except:
        print("Unable to read dataframe!")
        exit()
    return df

if __name__ == "__main__":
    main()