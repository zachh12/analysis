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
    #files = ["data/fl_det_1.csv", "data/fl_det_2.csv", "data/fl_det_3.csv", "data/fl_det_4.csv", "data/fl_det_5.csv"]
    files = ["data/fl_stored_1.csv", "data/fl_stored_2.csv", "data/fl_stored_4.csv", "data/fl_stored_5.csv"]
    files = ["data/oil_1.csv", "data/oil_2.csv", "data/fl_det_1.csv"]
    #files = ["data/fl_det_1.csv", "data/fl_stored_1.csv", "data/oil_1.csv"]
    #files = ['data/fl_det_5.csv', 'data/fl_stored_5.csv', 'data/fl_det_4.csv', 'data/fl_stored_4.csv']
    #files = ['data/fl_stored_1.csv', 'data/fl_det_1.csv','data/fl_stored_2.csv', 'data/fl_det_2.csv', 'data/fl_det_3.csv','data/fl_stored_4.csv', 'data/fl_det_4.csv','data/fl_stored_5.csv', 'data/fl_det_5.csv']
    #files = ['data/fl_det_5.csv', 'data/fl_stored_1.csv', 'data/oil_1.csv']
    means = []
    stds = []
    amps = []
    wavelength, intensity = read_file(files[0])
    plt.scatter(wavelength, intensity, s=1)
    plt.title("Mineral Oil Fluorescence")
    plt.ylabel("Intensity (arb)")
    plt.xlabel("Wavelength (nm)")
    plt.ylim(-.01, .04)
    plt.show()
    exit()
    if mode == 0:
        for file in files:

            wavelength, intensity = read_file(file)
            params = FitFlor(wavelength, intensity)
            df.loc[len(df)] = wavelength, intensity, params
            plt.scatter(wavelength, intensity, alpha=1, s=1)
            plt.plot(wavelength,testmodal(wavelength,*params), color="r")
            means.append([params[0], params[3], params[6], params[9], params[12]])
            stds.append([params[1], params[4], params[7], params[10], params[13]])
            amps.append([params[2], params[5], params[8], params[11], params[14]])

        #plt.ylim(-.02, .06)
        #plt.xlim(300, 500)
        plt.show()
        #exit()
        #Plot
        mean0, mean1, mean2, mean3, mean4 = [], [], [], [], []
        std0, std1, std2, std3, std4= [], [], [], [], []
        amp0, amp1, amp2, amp3, amp4= [], [], [], [], []
        print(means)
        #exit()
        for i in range(0, 2):
            mean0.append(means[i][0])
            mean1.append(means[i][1])
            mean2.append(means[i][2])
            mean3.append(means[i][3])
            mean4.append(means[i][4])
            std0.append(stds[i][0])
            std1.append(stds[i][1])
            std2.append(stds[i][2])
            std3.append(stds[i][3])
            std4.append(stds[i][4])
            amp0.append(amps[i][0])
            amp1.append(amps[i][1])
            amp2.append(amps[i][2])
            amp3.append(amps[i][3])
            amp4.append(amps[i][4])
        print(np.mean(mean0))
        params = [np.mean(mean0), np.mean(std0),np.mean(amp0), np.mean(mean1), np.mean(std1), np.mean(amp1), np.mean(mean2), np.mean(std2), np.mean(amp2), np.mean(mean3), np.mean(std3), np.mean(amp3), np.mean(amp4), np.mean(mean4), np.mean(std4)]
            #, np.mean(mean4), np.mean(std4), np.mean(amp4)
        wavelength = np.linspace(300, 600, 10000)
        print(params)
        #exit()
        plt.plot(wavelength,testmodal(wavelength,*params))#, color='r')
        plt.show()
        #exit()

        #Save
        np.savetxt( "Params_stored.txt", params)
        df.to_hdf("Stored.h5", key='data')
        #print(params)
        plt.show()
        #plotFlor(df)

    elif mode == 1:
        for file in files:
            wavelength, intensity = read_file(file)
            params = FitUV(wavelength, intensity)
            df.loc[len(df)] = wavelength, intensity, params
        #Plot


def FitFlor(wavelength, intensity):

    #Bimodal Distribution
    #expected = (424, 14, .08, 444, 8, .05)
    #params, cov = curve_fit(bimodal,wavelength,intensity,expected)

    #Trimodal Distribution
    expected = (406, 10, .35, 430, 8, .5, 455, 12, .4, 483, 12, .21, 530, 12, .1)
    params, cov = curve_fit(testmodal,wavelength,intensity,expected)

    return params

def plotFlor(df):
    for index, row in df.iterrows():
        wavelength, intensity, params = row[0], row[1], row[2]
        #plt.scatter(wavelength, intensity, s=1)
        plt.plot(wavelength,testmodal(wavelength,*params))#, color='r')
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

def testmodal(x,mu1,sigma1,A1,mu2,sigma2,A2, mu3,sigma3,A3, mu4,sigma4,A4, mu5,sigma5,A5):
    return gaussian(x,mu1,sigma1,A1)+gaussian(x,mu2,sigma2,A2)+gaussian(x,mu3,sigma3,A3)+gaussian(x,mu4,sigma4,A4)+gaussian(x,mu5,sigma5,A5)

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