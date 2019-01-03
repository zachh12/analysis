import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
#import siggen
#from siggen import PPC
#import dnest4.classic as dn
from scipy import stats
import statsmodels

chan = 626
delta = 564
delta2 = 0
tau = 159
def main():
    #Get Data
    df1 = getDataFrame("data/-1chan626data.h5")
    df2 = getDataFrame("data/OLD0-6chan626data.h5")
    df = df1.append(df2)
    #print(df2)
    #exit()
    #Remove bad wfs
    cut = df["training_id"] != 285985

    #Apply any cuts
    #cut = cut & (df['avse'] < 3) & (df['avse'] > 0)
    df = df[cut]
    #plt.hist(df['avse'])
    #plt.show()
    #exit()
    '''lens =  [30, 31, 25, 31, 31, 26]'''

    tune(df)
    plotFWHM(df)
    #plotCorrelation(df)

def plotFWHM(df):
    print(np.std(df['ecal']) * 2.35)
    tcorr = correct(df['ecal'], df['drift_time'], tau)
    print(np.std(tcorr) * 2.35)
    lcorr = correct(df['ecal'], df['hole_drift_length'], delta)
    print(np.std(lcorr) * 2.35)
    plt.figure(1)
    plt.hist(df["ecal"]-np.mean(df["ecal"]), alpha=.5, histtype=u'step', color='red', density=True, linewidth=4, label="ecal FWHM: " + str(round(np.std(df["ecal"]) * 2.35, 4)))
    plt.xlim(-6, 5)
    plt.legend()
    plt.xlabel("Energy Centered Around Mean (keV)")
    plt.title("ECal")
    plt.ylabel("Energy Density")
    #plt.figure(2)
    plt.hist(tcorr-np.mean(tcorr), alpha=.5, histtype=u'step', color='blue', density=True, linewidth=4, label="Time Correction: " + str(round(np.std(tcorr) * 2.35, 4)))
    plt.legend()
    plt.xlim(-6, 5)
    plt.xlabel("Energy Centered Around Mean (keV)")
    plt.title("Time Correction")
    plt.ylabel("Energy Density")
    #plt.figure(3)
    plt.hist(lcorr-np.mean(lcorr), alpha=.5, histtype=u'step', color='green', density=True, linewidth=4, label="Length Correction: " + str(round(np.std(lcorr) * 2.35, 4)))
    plt.legend()
    plt.title("Length Correction")
    plt.xlim(-6, 5)
    plt.xlabel("Energy Centered Around Mean (keV)")
    plt.ylabel("Energy Density")
    plt.show()

def plotCorrelation(df):
    '''slope, intercept, r_value, p_value, std_err = stats.linregress(df['drift_time'], df['hole_drift_length'])
    plt.scatter(df["drift_time"], df["hole_drift_length"], color='purple', label="R-Value: " + str(round(r_value, 2)))
    plt.legend()
    plt.title("Drift Time vs Hole Drift Length")
    plt.xlabel("Drift Time (10 ns)")
    plt.ylabel("Hole Drift Length (mm)")
    #plt.scatter(df["hole_drift_length"], df["drift_time"])
    plt.show()'''

    plt.figure(1)
    plt.subplot(1,2,1)
    stdDT = (df['drift_time'] - np.mean(df['drift_time']))/np.std(df['drift_time'])
    stdDL = (df['hole_drift_length'] - np.mean(df['hole_drift_length']))/np.std(df['hole_drift_length'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['ecal'], stdDT)
    #plt.scatter(df['ecal'], stdDT, color='purple', label="R-Value: " + str(round(r_value, 2)))
    sns.regplot(x=df['ecal'],y=stdDT, color ='blue', label="R-Value: "+ str(round(r_value, 2)))
    plt.xlabel("Energy (keV)")
    plt.ylabel("Normalized Drift Time")
    plt.title("ECal vs Drift Time")
    plt.xlim(2608, 2618)
    #plt.ylim(-.1, 1.1)
    plt.legend()

    plt.subplot(1,2,2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['ecal'], stdDL)
    #plt.scatter(df['ecal'], stdDL, color='green', label="R-Value: " + str(round(r_value, 2)))
    sns.regplot(df['ecal'], stdDL, color ='green', label="R-Value: "+ str(round(r_value, 2)))
    plt.xlabel("Energy (keV)")
    plt.ylabel("Normalized Hole Drift Length")
    plt.title("ECal vs Drift Length")
    plt.legend()
    plt.xlim(2608, 2618)
    #plt.ylim(-.1, 1.1)
    plt.show()

def tune(df):
    check = np.linspace(0, 700, 600)
    check2 = np.linspace(0, 700, 300)
    check3 = np.linspace(-200, 200, 200)
    rez, rez2, rez3 = [], [], []
    hdelta = []
    edelta = []
    for var in check:
        rez.append(np.std(df['ecal'] * np.exp((df['hole_drift_length'])*(var/10000000)))*2.35)
    for var in check:
        rez2.append(np.std(df['ecal'] * np.exp(df['drift_time'] * (var/10000000))) * 2.35)
    for var in check2:
        for var2 in check3:
            hdelta.append(var)
            edelta.append(var2)
    global delta
    delta = check[np.argmin(rez)]
    global tau
    tau = check[np.argmin(rez2)]

def correct(energy, var, constant):
    return (energy * np.exp(var * (constant/10000000)))

def getDataFrame(name):
    try:
        df = pd.read_hdf(name, key='data')
    except:
        print("Unable to read dataframe!")
        exit()
    return df

if __name__ == "__main__":
    main()