import pandas as pd
from siggen import PPC
from waffle.processing import *
import numpy as np
import dnest4.classic as dn4
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

def main():
    sample = np.loadtxt("Posteriors/1.txt")
    r, z = [], []
    for i in range(0,9):
        r.append(sample[i][0])
        z.append(sample[i][1])
    slope, intercept, r_value, p_value, std_err = stats.linregress(r, z)
    print("Slope: ", slope)
    x = np.linspace(0, 50)
    y = x * slope + intercept
    regression = plt.plot(x, y, label="R: " + str(('%.5f'%(r_value))))
    sample = np.loadtxt("Posteriors/2.txt")
    r, z = [], []
    for i in range(0,5):
        r.append(sample[i][0])
        z.append(sample[i][1])
    print(r,z)
    slope, intercept, r_value, p_value, std_err = stats.linregress(r, z)
    print("Slope: ", slope)
    x = np.linspace(10, 50)
    y = x * slope + intercept
    regression = plt.plot(x, y, label="R: " + str(('%.5f'%(r_value))))

    plt.show()
if __name__ == "__main__":
    main()