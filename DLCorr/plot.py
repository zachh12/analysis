import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4.classic as dn
from scipy import stats


chan = 626
def main():
    plots()

def plots():
    df = getDataFrame()
    plt.hist(df['ecal'])
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