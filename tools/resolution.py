import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygama
#from waffle.processing import *
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
from scipy import stats
import os
import h5py

def main():

    ecal = np.loadtxt("ecal.txt")
    #plt.hist(ecal, bins=1300)
    #plt.xlim(1300, 1500)
    #plt.show()
    #exit()
    k40, co60 = [], []
    for num in ecal:
        if ((num > 1435) & (num < 1475)):
            k40.append(num)
        elif ((num > 1300) & (num < 1370)):
            co60.append(num)
        #elif ((num ))
    print(len(k40))
    plt.hist(k40, bins=30)
    #plt.hist(co60)
    print(np.std(k40)*2.35)
    plt.show()
    #print(np.std(co60)*2.35) 

if __name__ == '__main__':
    main()