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

    plt.figure(figsize=(10,5))
    params_det = np.loadtxt("Params_det.txt")
    df_det = pd.read_hdf("det.h5", key="data")
    params_stored = np.loadtxt("Params_stored.txt")
    df_stored = pd.read_hdf("Stored.h5", key="data")

    wavelength = np.linspace(300, 600, 40)
    wavelength2 = np.linspace(302, 605, 40)
    full = np.linspace(300, 600, 300)
    #wavelength2 = wavelength

    model = []
    model2 = []
    errorDet, errorStore = np.array([]), np.array([])
    errorDetLow, errorDetHigh = [], []
    errorStoreLow, errorStoreHigh = [], []
    for i in range(0, 5):
        model.append(testmodal(wavelength,*df_det['fit_info'][i]))
        model2.append(testmodal(wavelength2,*df_stored['fit_info'][i]))

    for i in range(0, len(wavelength)):
        test = np.empty(2)
        test.fill(wavelength[i])
        error = [model[0][i], model[1][i], model[2][i], model[3][i], model[4][i]]
        errorDetLow.append(np.mean(error) - np.amin(error))
        errorDetHigh.append(np.amax(error) - np.mean(error))
        test.fill(wavelength2[i])
        error2 = [model2[0][i], model2[1][i], model2[2][i], model2[3][i], model2[4][i]]
        errorStoreLow.append(np.mean(error2) - np.amin(error2))
        errorStoreHigh.append(np.amax(error2) - np.mean(error2))

    plt.plot(full,testmodal(full,*params_det), alpha=.9, color="r", label="Detector Sample", linewidth=4)
    plt.errorbar(wavelength, testmodal(wavelength,*params_det), xerr=0, yerr=[errorDetLow, errorDetHigh], ls='none', color='r')
    plt.errorbar(wavelength2, testmodal(wavelength2,*params_stored), xerr=0, yerr=[errorStoreLow, errorStoreHigh], ls='none', color='b')
    plt.plot(full,testmodal(full,*params_stored), alpha=.9, color="b", label="Stored Sample", linewidth=4)
    plt.xlim(375, 520)
    #plt.ylim(.3, .7)
    plt.xlabel("Wavelength (nm)")# \n More info")
    plt.ylabel("Intensity (arb)")
    plt.legend()

    plt.savefig("Error.png", bbox_inches='tight')
    plt.show()


def plotFlor(df):
    for index, row in df.iterrows():
        wavelength, intensity, params = row[0], row[1], row[2]
        #plt.scatter(wavelength, intensity, s=1)
        plt.plot(wavelength,testmodal(wavelength,*params))#, color='r')
        plt.xlim(300, 700)
    plt.show()



def fit(x, y):
    init_vals = [x[np.argmax(y)], np.std(y), np.amax(y)]
    best_vals, covar = curve_fit(gaussian, x, y, p0=init_vals)
    #if plot:
    #    label = "Wavelength: " + str('%.1f'%best_vals[0]) + "\nIntensity: " + str('%.3f'%best_vals[2])
    #    plt.plot(x, y, color=color[c], label=label)
    #    plt.legend()
    return best_vals[0], best_vals[1], best_vals[2]

def testmodal(x,mu1,sigma1,A1,mu2,sigma2,A2, mu3,sigma3,A3, mu4,sigma4,A4, mu5,sigma5,A5):
    return gaussian(x,mu1,sigma1,A1)+gaussian(x,mu2,sigma2,A2)+gaussian(x,mu3,sigma3,A3)+gaussian(x,mu4,sigma4,A4)+gaussian(x,mu5,sigma5,A5)



def gaussian(x, mu, sigma, amp):
    return amp * exp(-(x-mu)**2 / sigma)



if __name__ == "__main__":
    main()