from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import optimize
from scipy import asarray as ar,exp
from scipy.stats import norm
from scipy.optimize import curve_fit

def main():
    plotAll = False
    #UV("uv_null_1.csv", "uv_sc_1.csv", plotAll, plot=True)
    #UV("uv_null_2.csv", "uv_sc_2.csv", plotAll, plot=True)
    Flor("flor_avg.csv")

    if plotAll:
        plt.xlim(400, 600)
        plt.show()

def Flor(file):
    wavelength, intensity = read_file(file)
    plt.scatter(wavelength, intensity, s=1)
    #plt.show()
    #expected = (wavelength[np.argmax(intensity)], np.std(intensity), np.amax(intensity), wavelength[np.argmax(intensity)]+20, np.std(intensity), np.amax(intensity)-.1)
    expected = (444, 14, .08, 424, 8, .08)
    params, cov = curve_fit(bimodal,wavelength,intensity,expected)
    print(params[0], params[3])
    plt.plot(wavelength,bimodal(wavelength,*params), color='r')
    plt.show()


def UV(file1, file2, plotAll=False, plot=False):
    wavelength, intensity = read_file(file1)
    wavelength2, intensity2 = read_file(file2)
    mean, std, amp = fit(wavelength, intensity, plot, 0)
    mean2, std, amp2 = fit(wavelength2, intensity2, plot, 1)
    print("Transmittance @", '%.3f'%mean + ":", '%.3f'%(amp2 / amp))

    if ((plot) & (not plotAll)):
        plt.xlim(400, 550)
        plt.show()

def fit(x, y, plot, c):
    color = ['b', 'g']
    init_vals = [x[np.argmax(y)], np.std(y), np.amax(y)]
    best_vals, covar = curve_fit(gaussian, x, y, p0=init_vals)
    x = np.linspace(np.min(x), np.max(x), 1000)
    y = gaussian(x, best_vals[0], best_vals[1], best_vals[2])
    if plot:
        label = "Wavelength: " + str('%.1f'%best_vals[0]) + "\nIntensity: " + str('%.3f'%best_vals[2])
        plt.plot(x, y, color=color[c], label=label)
        plt.legend()
    return best_vals[0], best_vals[1], best_vals[2]

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