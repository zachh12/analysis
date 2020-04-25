import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4.classic as dn
from scipy import stats
from scipy.stats import norm

bad = [26384.0, 24579.0]
chan = 626
tau = 753
tau2 = 228

def main():
    plots()

def plots():

    df = getDataFrame()

    #print(df['waveform'])
    #exit()

    cut = (df['ecal'] > 2000)
    df = df[cut]
    print(len(df))
    plt.hist(df['ecal'], histtype='step', lw=2,color='r', bins=15)

    plt.show()
    #exit()
    tune(df)
    ecorr, ecorrt = corr(df)
    FWHM(df, ecorr, ecorrt)

    plt.show()
def tune(df):
    check = np.linspace(0, 700, 200)
    check2 = np.linspace(0, 700, 200)
    check3 = np.linspace(-200, 200, 200)
    rez, rez2, rez3 = [], [], []
    hlambda = []
    elambda = []
    for var in check:
        rez.append(np.std(df['ecal'] * np.exp((df['hole_drift_length'])*(var/10000000)))*2.35)
    for var in check:
        rez2.append(np.std(df['ecal'] * np.exp(df['drift_time'] * (var/10000000))) * 2.35)
    #for var in check2:
    #    for var2 in check3:
    #        hlambda.append(var)
    ##        elambda.append(var2)
    #        rez3.append(2.35 * np.std(df['ecal'] * np.exp((df['hole_drift_length'] * (var/10000000)) + df['electron_drift_length'] * (var2/10000000))))
    #tau = check[np.argmin(rez)]
    #tau2 = check[np.argmin(rez2)]
    print("None: ", np.min(np.std(df['ecal'])*2.35))
    print("Time: ", np.min(rez2))
    print("Length: ", np.min(rez))
    #print("Double: ", np.min(rez3))

    plt.plot(rez)
    plt.title("FWHM Minimization")
    plt.xlabel("λ Parameter")
    plt.ylabel("FWHM @2614 keV")
    #minimum = np.argmin(rez3)
    print("Time Arg: ",check[np.argmin(rez2)])
    #print(hlambda[minimum], elambda[minimum])
    plt.show()
    #exit()

def corr(df):
    energy = df['ecal'] - np.mean(df['ecal'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['hole_drift_length'], energy)
    print("Slope: ", slope)
    x = np.linspace(10, 50)
    y = x * slope + intercept
    regression = plt.plot(x, y, label="R: " + str(('%.5f'%(r_value))))
    plt.legend(loc='upper right')
    plt.scatter(df['hole_drift_length'], energy)
    plt.xlabel("Drift Length (mm)")
    plt.ylabel("Energy (arb)")
    plt.ylim(-4, 4)
    plt.title("Drift Length vs Energy @2614 keV Peak")
    plt.figure(2)
    ecorr = df['ecal'] * np.exp(df['hole_drift_length']*(tau/10000000))
    ecorrt = df['ecal'] * np.exp(df['drift_time'] * (tau2/10000000))
    ecorr = ecorr - np.mean(ecorr)
    plt.scatter(df['hole_drift_length'], ecorr - np.mean(ecorr))
    plt.title("Drift Length vs Length Corrected Energy")
    plt.xlabel("Drift Length (mm)")
    plt.ylabel("Energy (arb)")
    plt.ylim(-4, 4)
    plt.show()
    #exit()
    print("None: ", np.std(df['ecal'])*2.35)
    print("Time: ", np.std(ecorrt)*2.35)
    print("Length: ", np.std(ecorr)*2.35)
    plt.figure(100)
    plt.hist(df['ecal'] - np.mean(df['ecal']), alpha=.6, bins=20)
    plt.hist(ecorrt - np.mean(ecorrt), alpha=.5, bins=20)
    plt.hist(ecorr - np.mean(ecorr), alpha=.5, bins=12)
    plt.show()
    return ecorr, ecorrt

def FWHM(df, ecorr, ecorrt):
    energy = df.ecal - np.mean(df.ecal)
    ecorr = ecorr - np.mean(ecorr)
    ecorrt = ecorrt - np.mean(ecorrt)
    bins2 = np.linspace(-4, 4, 400)

    plt.figure(1)
    plt.title("Length-Based Corrected Energy", fontsize=20)
    plt.xlabel("Energy Centered Around Mean (keV)")
    (mu, sigma) = norm.fit(ecorr)
    n, bins, patches = plt.hist(ecorr, 12, lw=4, histtype="step",normed=1, color='gray', alpha=1)
    y = mlab.normpdf( bins2, mu, sigma)
    l = plt.plot(bins2, y, 'r', linewidth=2, linestyle='--',color='mediumblue', label='Length-Based Correction, FWHM: 2.70 keV')
    plt.legend(shadow=True, fancybox=True, loc='upper right', prop={'size': 10},)
    plt.xlim(-4, 4)

    plt.figure(2)
    plt.title("Time-Based Corrected Energy", fontsize=20)
    plt.xlabel("Energy Centered Around Mean (keV)")
    (mu, sigma) = norm.fit(ecorrt)
    n, bins, patches = plt.hist(ecorrt, 12, lw=4, histtype="step", normed=1, color='gray', alpha=1)
    y = mlab.normpdf( bins2, mu, sigma)
    l = plt.plot(bins2, y, 'g', linewidth=2, linestyle='--', color='mediumblue', label='Time-Based Correction, FWHM: 3.17 keV')
    plt.legend(shadow=True, fancybox=True, loc='upper right', prop={'size': 10},)
    plt.xlim(-4, 4)

    plt.figure(3)
    plt.title("Measured Energy Distribution", fontsize=20)
    plt.xlabel("Energy Centered Around Mean (keV)")
    (mu, sigma) = norm.fit(energy)
    n, bins, patches = plt.hist(energy, 12, lw=4, histtype="step", normed=1, color='gray', alpha=1)
    y = mlab.normpdf( bins2, mu, sigma)
    l = plt.plot(bins2, y, 'b', linewidth=2, linestyle='--', color='mediumblue', label='No Correction, FWHM: 3.89 keV')
    plt.xlim(-4, 4)
    plt.legend(shadow=True, fancybox=True, loc='upper right', prop={'size': 10},)
    #plt.title("Energy Resolution @2614 keV")
    #plt.xlabel("Energy Centered at Mean")
    plt.show()
def getDataFrame():
    name = "data/trapC_chan626data.h5"
    #name = "data/chan"
    try:
        df = pd.read_hdf(name, key='data')
    except:
        print("Unable to read dataframe!")
        exit()
    return df

if __name__ == "__main__":
    main()