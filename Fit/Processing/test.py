import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4

def func(x):
	return 2614*np.exp(-x/20000)
def plot_train():
    length = np.arange(1, 50, 1)
    energy = func(length)

    plt.plot(energy)
    plt.show()

def main():
    #wfs = np.arange(0, 41, 1)

    #consistency(wfs)
    #energy(wfs)
    plot_train()


if __name__ == "__main__":
    main()