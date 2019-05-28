from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import optimize
from scipy import asarray as ar,exp
from scipy.stats import norm
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns; sns.set()

#TODO Error Bars, Result summary for each type

def main():

    params_det = np.loadtxt("Params_det.txt")
    df_det = pd.read_hdf("det.h5", key="data")
    params_stored = np.loadtxt("Params_stored.txt")
    df_stored = pd.read_hdf("Stored.h5", key="data")

    wavelength = np.linspace(300, 600, 40)
    wavelength2 = np.linspace(302, 605, 40)
    full = np.linspace(300, 600, 300)

    model, model2 = [], []
    errorDetLow, errorDetHigh = [], []
    errorStoreLow, errorStoreHigh = [], []
    for i in range(0, 5):
        model.append(pentmodal(wavelength,*df_det['fit_info'][i]))
        model2.append(pentmodal(wavelength2,*df_stored['fit_info'][i]))

    for i in range(0, len(wavelength)):
        error = [model[0][i], model[1][i], model[2][i], model[3][i], model[4][i]]
        errorDetLow.append(np.mean(error) - np.amin(error))
        errorDetHigh.append(np.amax(error) - np.mean(error))
        error2 = [model2[0][i], model2[1][i], model2[2][i], model2[3][i], model2[4][i]]
        errorStoreLow.append(np.mean(error2) - np.amin(error2))
        errorStoreHigh.append(np.amax(error2) - np.mean(error2))

    plt.figure(figsize=(12,7))
    plt.plot(full, pentmodal(full,*params_det), alpha=1, color="r", label="Detector Sample", linewidth=2)
    plt.errorbar(wavelength, pentmodal(wavelength,*params_det), xerr=2, yerr=[errorDetLow, errorDetHigh], fmt='.', color='r')
    plt.plot(full, pentmodal(full,*params_stored), alpha=1, color="b", label="Stored Sample", linewidth=2)
    plt.errorbar(wavelength2, pentmodal(wavelength2,*params_stored), xerr=2, yerr=[errorStoreLow, errorStoreHigh], fmt='.', color='b')

    plt.title("PROSPECT Scintillator Emission Spectrum")
    txt = "Thorlabs CCS200 spectrophotometer, Hamamatsu Model XXX Xe flash lamp, 1cm x 1cm quartz cuvette, spectrum at taken at 90 degrees detector-source angle, 10,000us integration time,  etc" 
    plt.xlim(350, 550)
    plt.xlabel("Wavelength (nm) \n")
    plt.ylabel("Intensity (arb)")
    plt.legend()

    plt.savefig("Error.png", bbox_inches='tight')
    plt.show()


def pentmodal(x,mu1,sigma1,A1,mu2,sigma2,A2, mu3,sigma3,A3, mu4,sigma4,A4, mu5,sigma5,A5):
    return gaussian(x,mu1,sigma1,A1)+gaussian(x,mu2,sigma2,A2)+gaussian(x,mu3,sigma3,A3)+gaussian(x,mu4,sigma4,A4)+gaussian(x,mu5,sigma5,A5)

def gaussian(x, mu, sigma, amp):
    return amp * exp(-(x-mu)**2 / sigma)

if __name__ == "__main__":
    main()