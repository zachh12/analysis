import pandas as pd
from siggen import PPC
from waffle.processing import *
import numpy as np
import dnest4.classic as dn4
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
sns.set()

chan = 692
const = 350

def main():
    df = pd.read_hdf("data/chan" + str(chan) + "data.h5", key='data')
    #df = df[df['ecal'] > 2000]
    print(2.35*np.std(df['ecal']))
    #print(2.35*np.std(df['trap_max']))
    #print(2.35*np.std(df['trap_ft']))
    #print(2.35*np.std(df['fitE']))
    #exit()
    #exit()
    df['sim_hole_drift_time'] = np.float64(df['sim_hole_drift_time'])
    df['hole_drift_length'] = np.float64(df['hole_drift_length'])
    df['sim_electron_drift_time'] = np.float64(df['sim_electron_drift_time'])

    plt.figure(1)
    plt.subplot(1,2,1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['drift_time'], df['trap_max'])
    plt.scatter( df['drift_time'] * 10, df['trap_max'], color='green', label="R: " + str(('%.3f'%(r_value))))
    plt.ylabel("Energy (keV)")
    plt.xlabel("Calculated Drift Time (ns)")
    plt.title("Energy vs Calculated Drift Time")
    plt.legend(loc='upper right')
    plt.subplot(1,2,2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['sim_hole_drift_time'], df['trap_max'])
    plt.scatter(df['sim_hole_drift_time'], df['trap_max'], color='green',label="R: " + str(('%.3f'%(r_value))))
    plt.ylabel("Energy (keV)")
    plt.xlabel("Fitted Drift Time (ns)")
    plt.title("Energy vs Fitted Drift Time")
    plt.legend(loc='upper right')
    plt.show()
    exit()
    printFWHM(df)
    '''
    plt.figure(1)
    plt.scatter( df['sim_hole_drift_time'],df['drift_time']*10, color='purple')
    plt.xlabel("Fitted Drift Time (ns)")
    plt.ylabel("Calculated Drift Time (ns)")
    plt.title("Simulated vs Calculated Drift Time")
    #plt.figure(2)
    #plt.scatter( df['drift_time'],df['trap_ft'])
    #plt.figure(3)
    #plt.scatter(df['hole_drift_length'],df['trap_ft'] )
    plt.show()
    plt.hist(df['ecal'] - np.mean(df['ecal']), histtype=u'step')
    '''
    var = df['sim_hole_drift_time']
    optimize(df['trap_ft'], var)
    df['trap_ft'] = df['trap_ft'] * np.exp((var) * const/10000000)
    optimize(df['trap_max'], var)
    df['trap_max'] = df['trap_max'] * np.exp((var) * const/10000000)
    optimize(df['ecal'], var)
    df['ecal'] = df['ecal'] * np.exp(var * const/10000000)
    optimize(df['fitE'], var)
    df['fitE'] = df['fitE'] * np.exp(var * const/10000000)
    #plt.hist(df['ecal']-np.mean(df['ecal']),histtype=u'step')
    #plt.show()
    printFWHM(df)


def optimize(energy, variable):
    check = np.linspace(-2000, 2000, 1500)
    rez = []
    for var in check:
        rez.append(np.std(energy * np.exp((variable)*(var/10000000)))*2.35)

    global const
    const = check[np.argmin(rez)]
    #print(np.min(rez), const)

def printFWHM(df):
    print(np.std(df['trap_max']) * 2.35, np.std(df['trap_ft']) * 2.35,
        np.std(df['fitE']) * 2.35, np.std(df['ecal']) * 2.35)

def plotFWHM(df):
    plt.figure(1)
    dets = [0, 1, 2]
    raw = [3.2, 3.1, 3.0]
    opt = [3.0, 3.0, 2.5]
    plt.scatter(dets, raw, label="Raw")
    plt.scatter(dets, opt, label="Opt")
    plt.title("Energy Resolution")
    plt.legend(loc='upper right')
    plt.xlabel("Detector")
    plt.ylabel("FWHM @ 2615 keV")
    plt.xticks(np.arange(3), ('626', '672', '692'))
    plt.axhline(y=3.0, linewidth=.5, color='r')
    plt.show()

if __name__ == '__main__':
    main()