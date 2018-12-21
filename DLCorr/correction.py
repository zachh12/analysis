import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4.classic as dn
from scipy import stats
from scipy.stats import norm

chan = 626
delta = 753
tau = 228

def main():
    df = getDataFrame()
    #cut = (df['avse'] > 0) & (df['avse'] < 3)
    #df = df[cut]
    plt.hist(df['drift_time'] * 10, bins=20)
    plt.show()
    tune(df)
    print(np.std(df['ecal']) * 2.35)
    tcorr = correct(df['ecal'], df['drift_time'], tau)
    print(np.std(tcorr) * 2.35)
    lcorr = correct(df['ecal'], df['hole_drift_length'], delta)
    print(np.std(lcorr) * 2.35)

def tune(df):
    check = np.linspace(0, 700, 200)
    check2 = np.linspace(0, 700, 200)
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
    delta = check[np.argmin(rez)]
    tau = check[np.argmin(rez2)]

def correct(energy, var, constant):
    return (energy * np.exp(var * (constant/10000000)))

def getDataFrame():
    name = "data/chan626data.h5"
    try:
        df = pd.read_hdf(name, key='data')
    except:
        print("Unable to read dataframe!")
        exit()
    return df

if __name__ == "__main__":
    main()