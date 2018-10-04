import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4.classic as dn
from scipy import stats

def calibrate(list):

    df = np.load("../training_data/chan692_2614wfs.npz")
    df2 = pd.read_hdf("../training_data/training_set.h5")
    indexes = []
    times = []
    Energy = []
    for i in list:
        indexes.append((df['wfs'][i]).training_set_index)
        dir = "../chan692_2614wfs/wf" + str(int(i)) + "/posterior_sample.txt"
        post = np.loadtxt(dir)
    for index in indexes:
        Energy.append(df2['ecal'][index])
        times.append(df2['drift_time'][index])
    return Energy, times

def getDriftLength(det, r, theta, z):
    wf = det.GetWaveform(r, theta, z)
    hpath = det.siggenInst.GetPath(1)

    #r, hpath[0] ||| angle, hpath[1] ||| z, hpath[2]
    x = hpath[0]*np.cos(hpath[1])
    y = hpath[0]*np.sin(hpath[1])
    z = hpath[2]

    length = 0
    for k in range(1, len(x)):
        length += np.sqrt((x[k]-x[k-1])**2 + 
            (y[k]-y[k-1])**2 + (z[k]-z[k-1])**2)

    return length


def energy(list, energy, times):
    r = []
    z = []
    theta = []
    drift_length = []
    det = PPC("config_files/p1_new.config")
    for i in list:
        dir = "../chan692_2614wfs/wf" + str(int(i)) + "/posterior_sample.txt"
        post = np.loadtxt(dir)
        rtemp, ztemp, thetatemp = [], [], []
        for sample in post:
            try:
                rtemp.append(sample[0])
                ztemp.append(sample[1])
                thetatemp.append(sample[2])
            except:
                rtemp.append(post[0])
                ztemp.append(post[1])
                thetatemp.append(post[2])
        r.append(rtemp[0])
        z.append(ztemp[0])
        theta.append(thetatemp[0])
    for i in range(0, len(r)):
        length = getDriftLength(det, r[i], theta[i], z[i])
        drift_length.append(length)
    drift_length = np.array(drift_length)

    r = np.array(r)
    slope, intercept, r_value, p_value, std_err = stats.linregress(drift_length, energy)

    plt.figure(1)
    plt.scatter(times, drift_length)
    plt.ylabel("Drift Length (mm)")
    plt.xlabel("Drift Time (us)")
    plt.figure(2)
    plt.scatter(np.square(r), z)
    plt.xlabel("Radius^2 (mm)")
    plt.ylabel("Z (mm)")
    plt.figure(3)
    plt.scatter(drift_length, energy)
    x = np.linspace(1, 300)
    y = x * slope + intercept
    plt.plot(x, y)
    plt.xlim(.9*np.min(drift_length), 1.1*np.max(drift_length))
    plt.xlabel("Drift Length (mm)")
    plt.ylabel("Energy (keV)")
    #MJD Resolution = 2.3(3.1?)
    print("Resolution: ", np.std(energy)*2.35)
    plt.show()

def main():

    wfs = np.arange(0, 8, 1)
    Energy, times = calibrate(wfs)
    energy(wfs, Energy, times)

if __name__ == "__main__":
    main()