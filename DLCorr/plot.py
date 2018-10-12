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

    df = df[df['drift_length'] > 60]
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['drift_length'], df['ecal'])
    x = np.linspace(1, 300)
    y = x * slope + intercept
    regression = plt.plot(x, y, label="R: " + str(('%.5f'%(r_value))))
    plt.legend(loc='upper right')
    plt.xlim(.7*np.min(df['drift_length']), 1.1*np.max(df['drift_length']))
    plt.scatter(df['drift_length'], df['ecal'])
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