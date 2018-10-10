import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4.classic as dn
from scipy import stats
def calibrate():

    df = np.load("../training_data/chan626_2614wfs.npz")
    df2 = pd.read_hdf("../training_data/training_set.h5")
    lists = np.arange(0, 59)
    exclude= [16]
    valid = np.setdiff1d(lists,exclude)
    lists = valid
    indexes = []
    times = []
    Energy = []
    for i in lists:
        indexes.append((df['wfs'][i]).training_set_index)
        #dir = "../chan692_8wfs/wf" + str(int(i)) + "/posterior_sample.txt"
        #post = np.loadtxt(dir)
    k = 0
    for index in indexes:
        if (index == 194754):
            continue;
        Energy.append(df2['ecal'][index])
        times.append(df2['drift_time'][index])
        k += 1
    print("Resolution: ", np.std(Energy)*2.35)
    plt.hist(Energy, bins=40)
    plt.show()



def main():
    calibrate()

if __name__ == "__main__":
    main()