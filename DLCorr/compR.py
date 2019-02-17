import pandas as pd
from siggen import PPC
from waffle.processing import *
import numpy as np
import dnest4.classic as dn4
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

chan = 626
const = 350

def main():
    df = pd.read_hdf("data/chan" + str(chan) + "data.h5", key='data')
    plt.hist(df['trap_ft'])
    plt.show()
    exit()
    df['sim_hole_drift_time'] = np.float64(df['sim_hole_drift_time'])
    df['hole_drift_length'] = np.float64(df['hole_drift_length'])
    df['sim_electron_drift_time'] = np.float64(df['sim_electron_drift_time'])

    printFWHM(df)

    var = df['drift_time']
    optimize(df['trap_ft'], var)
    df['trap_ft'] = df['trap_ft'] * np.exp((var) * const/10000000)
    optimize(df['trap_max'], var)
    df['trap_max'] = df['trap_max'] * np.exp((var) * const/10000000)
    optimize(df['ecal'], var)
    df['ecal'] = df['ecal'] * np.exp(var * const/10000000)
    optimize(df['fitE'], var)
    df['fitE'] = df['fitE'] * np.exp(var * const/10000000)

    printFWHM(df)


def optimize(energy, variable):
    check = np.linspace(1, 2000, 1500)
    rez = []
    for var in check:
        rez.append(np.std(energy * np.exp((variable)*(var/10000000)))*2.35)

    global const
    const = check[np.argmin(rez)]
    print(np.min(rez), const)

def printFWHM(df):
    print(np.std(df['trap_max']) * 2.35, np.std(df['trap_ft']) * 2.35,
        np.std(df['fitE']) * 2.35, np.std(df['ecal']) * 2.35)
if __name__ == '__main__':
    main()