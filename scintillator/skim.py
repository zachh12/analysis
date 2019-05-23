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
    cols = ["wavelength", "intensity", "fit_info"]
    df = pd.DataFrame(columns=cols)

    # 0 == Flor, 1 == UV
    mode = 0
    files = ["data/flor_avg.csv"]#, "data/flor.csv"]
    if mode == 0:
        for file in files:
            wavelength, intensity = read_file(file)
            params = FitFlor(wavelength, intensity)
            df.loc[len(df)] = wavelength, intensity, params
        #Plot
        plotFlor(df)

    elif mode == 1:
        for file in files:
            wavelength, intensity = read_file(file)
            params = FitUV(wavelength, intensity)
            df.loc[len(df)] = wavelength, intensity, params
        #Plot



    #UV("data/uv_null_1.csv", "data/uv_sc_1.csv", plotAll, plot=True)
    #UV("data/uv_null_2.csv", "data/uv_sc_2.csv", plotAll, plot=True)
    #LoadFlor("data/flor_avg.csv")

def FitFlor(wavelength, intensity):

    #Bimodal Distribution
    #expected = (424, 14, .08, 444, 8, .05)
    #params, cov = curve_fit(bimodal,wavelength,intensity,expected)

    #Trimodal Distribution
    expected = (424, 14, .08, 444, 8, .05, 460, 12, .03)
    params, cov = curve_fit(trimodal,wavelength,intensity,expected)

    return params

def plotFlor(df):
    for index, row in df.iterrows():
        wavelength, intensity, params = row[0], row[1], row[2]
        plt.scatter(wavelength, intensity, s=1)
        plt.plot(wavelength,trimodal(wavelength,*params), color='r')
        plt.xlim(300, 700)
    plt.show()

def FitUV(wavelength, intensity, wavelength2, intensity2):
    mean, std, amp = fit(wavelength, intensity)
    params = [mean, std, amp]
    return params

def plotUV(wavelength, intensity, params):
    print("Todo")

def fit(x, y):
    init_vals = [x[np.argmax(y)], np.std(y), np.amax(y)]
    best_vals, covar = curve_fit(gaussian, x, y, p0=init_vals)
    #if plot:
    #    label = "Wavelength: " + str('%.1f'%best_vals[0]) + "\nIntensity: " + str('%.3f'%best_vals[2])
    #    plt.plot(x, y, color=color[c], label=label)
    #    plt.legend()
    return best_vals[0], best_vals[1], best_vals[2]

def trimodal(x,mu1,sigma1,A1,mu2,sigma2,A2, mu3,sigma3,A3):
    return gaussian(x,mu1,sigma1,A1)+gaussian(x,mu2,sigma2,A2)+gaussian(x,mu3,sigma3,A3)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gaussian(x,mu1,sigma1,A1)+gaussian(x,mu2,sigma2,A2)

def gaussian(x, mu, sigma, amp):
    return amp * exp(-(x-mu)**2 / sigma)

def read_file(name):
    raw = []
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        start = False
        for row in csv_reader:
            raw.append(row)

    line = 0
    raw_data = []
    for row in raw:
        if ((line > 33) & (line < len(raw)-1)):
            raw_data.append(row)
            line += 1
        else:
            line += 1
    wavelength, intensity = [], []
    for row in raw_data:

        x = row[0].split(';')
        wavelength.append(x[0])
        intensity.append(x[1])
    wavelength = np.array(wavelength, dtype=float)
    intensity = np.array(intensity, dtype=float)
    return wavelength, intensity

if __name__ == "__main__":
    main()