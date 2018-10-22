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
tau = 753
tau2 = 228
avse_dict = {
    0-1: 68.81137,
    1-2: 1123.88,
    2-3: 584.321,
    3-4: 304.616, #Smaller sample size
    4: -404.398 #uhm
}
bins = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
RawResolution = [4.21, 2.62, 4.69, 4.17, 3.28, 2.73, 4.23, 4.11]
Tresolution = [3.61, 2.61, 3.01, 2.51, 2.84, 1.71, 3.24, 4.05]
Lresolution = [3.82, 2.55, 2.34, 2.22, 2.40, 2.42, 3.41, 4.11]
Dresolution = [3.59, 2.54, 2.42, 2.07, 2.40, 1.68, 2.65, 3.85]
taus = [806, -193, 1227, 962, 537, 434, 885, 78]
def main():
    plots()

def plots():

    df = getDataFrame()
    '''cut = (df['avse'] > 1) & (df['avse'] < 3)
    df = df[cut]
    energy = df['ecal']
    dtdev = (df['drift_time'] - np.mean(df['drift_time'])) / np.std(df['drift_time'])
    slope, intercept, dt_r, p_value, std_err = stats.linregress(df['drift_time'], energy)
    dldev = (df['drift_length'] - np.mean(df['drift_length'])) / np.std(df['drift_length'])
    slope, intercept, dl_r, p_value, std_err = stats.linregress(df['drift_length'], energy)
    plt.subplot(1,2,2)
    ax = sns.regplot(x=df['ecal'], y=dldev, color='green', label='Drift Length' + " R: " + str(('%.5f'%(dl_r))))
    plt.ylabel("Drift Length Standard Deviation From Mean")
    plt.xlabel("Energy (keV)")
    plt.title("Energy vs Drift Length Standard Deviation")
    plt.xlim(2610, 2618)
    plt.ylim(-2.5, 2.5)
    plt.legend(fancybox=True, loc='upper right', prop={'size': 12}, markerscale=False)
    plt.subplot(1,2,1)
    ax2 = sns.regplot(x=df['ecal'], y=dtdev, color='purple', label='Drift Time' + " R: " + str(('%.5f'%(dt_r))))
    plt.ylabel("Drift Time Standard Deviation From Mean")
    plt.xlabel("Energy (keV)")
    plt.title("Energy vs Drift Time Standard Deviation")
    plt.xlim(2610, 2618)
    plt.ylim(-2.5, 2.5)
    plt.legend(fancybox=True, loc='upper right', prop={'size': 12}, markerscale=False)
    plt.show()

    exit()'''
    '''plt.hist(df['ecal'], histtype='step', lw=2, label='Energy', bins=15)
    cut = (df['avse'] > 1) & (df['avse'] < 4.5)
    df = df[cut]
    plt.hist(df['ecal'], histtype='step', lw=2,color='r', label='With AvsE Cut', bins=15)
    plt.xlabel("Energy (keV)")
    plt.legend()
    plt.show()
    exit()'''
    cut = (df['avse'] > 1) & (df['avse'] < 3)
    df = df[cut]
    tune(df)
    exit()
    ecorr, ecorrt = corr(df)
    #FWHM(df, ecorr, ecorrt)
    plt.show()
def tune(df):
    check = np.linspace(-500, 2000, 8000)
    check2 = np.linspace(-500, 500, 100)
    check3 = np.linspace(-500, 500, 100)
    rez, rez2, rez3 = [], [], []
    for var in check:
        rez.append(np.std(df['ecal'] * np.exp(df['drift_length']*(var/10000000)))*2.35)
    for var in check:
        rez2.append(np.std(df['ecal'] * np.exp(df['drift_time'] * (var/10000000))) * 2.35)
    '''for var in check2:
        for var2 in check3:
            rez3.append(2.35 * np.std(df['ecal'] * np.exp((df['drift_time'] * (var/10000000)) + df['drift_length'] * (var2/10000000))))
    tau = check[np.argmin(rez)]'''
    tau2 = check[np.argmin(rez2)]
    print("None: ", np.min(np.std(df['ecal'])*2.35))
    print("Time: ", np.min(rez2))
    print("Length: ", np.min(rez))
    #print("Double: ", np.min(rez3))
    #plt.figure(11)
    check = check/10000000
    check = 1/check
    plt.scatter(check, rez, s=18, color='firebrick')
    plt.title("FWHM Minimization")
    plt.xlabel("λ Parameter")
    plt.ylabel("FWHM @2614 keV")
    plt.xlim(3000,30000)
    plt.show()
    exit()
    print(tau, tau2)
    #exit()

def corr(df):
    energy = df['ecal'] - np.mean(df['ecal'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['drift_length'], energy)
    print("Slope: ", slope)
    x = np.linspace(10, 50)
    y = x * slope + intercept
    regression = plt.plot(x, y, label="R: " + str(('%.5f'%(r_value))))
    plt.legend(loc='upper right')
    plt.scatter(df['drift_length'], energy)
    plt.xlabel("Drift Length (mm)")
    plt.ylabel("Energy (arb)")
    plt.ylim(-4, 4)
    plt.title("Drift Length vs Energy @2614 keV Peak")
    plt.figure(2)
    ecorr = df['ecal'] * np.exp(df['drift_length']*(tau/10000000))
    ecorrt = df['ecal'] * np.exp(df['drift_time'] * (tau2/10000000))
    ecorr = ecorr - np.mean(ecorr)
    plt.scatter(df['drift_length'], ecorr - np.mean(ecorr))
    plt.title("Drift Length vs Length Corrected Energy")
    plt.xlabel("Drift Length (mm)")
    plt.ylabel("Energy (arb)")
    plt.ylim(-4, 4)
    plt.show()
    #exit()
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
    name = "data/chan" + str(chan) + "data.h5"
    try:
        df = pd.read_hdf(name, key='data')
    except:
        print("Unable to read dataframe!")
        exit()
    return df

if __name__ == "__main__":
    main()