import pandas as pd
from siggen import PPC
from waffle.processing import *
import numpy as np
import dnest4.classic as dn4
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os
tau = 159
def tune(df):
    check = np.linspace(-200, 1000, 300)

    rez, rez2, rez3 = [], [], []
    hdelta = []
    edelta = []
    for var in check:
        rez2.append(np.std(df['ecal'] * np.exp(df['drift_time'] * (var/10000000))) * 2.35)
    global tau
    tau = check[np.argmin(rez2)]

#wfList = np.load("data/0-6/chan626_2614wfs.npz")
    #trainingIdx = wfList['wfs'][idx].training_set_index
chanList = [580, 626, 672, 692]
for chan in chanList:
    trainingSet = pd.read_hdf("datarun11520-11524.h5")
    cut = trainingSet['channel'] == chan
    trainingSet = trainingSet[cut]
    tune(trainingSet)
    energy = trainingSet['ecal'] * np.exp(trainingSet['drift_time'] * (tau/10000000))
    print(np.std(trainingSet['ecal']) * 2.35, (np.std(energy) * 2.35))
    #print(len(energy))
    plt.hist(energy, alpha=.4)
plt.show()



