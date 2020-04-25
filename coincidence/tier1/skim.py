import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def main(run):
    count = 0
    if int(run) < 1776:
        df = pd.read_hdf("t1_run" + str(run) + ".h5", key="ORSIS3302DecoderForEnergy")
    else:
        df = pd.read_hdf("t1_run" + str(run) + ".h5", key="ORSIS3316WaveformDecoder")

    A, E, fit, lowA = [], [], [], []
    for i in range(0,15):
        A.append(maxCurr(df.iloc[i][6:]))
        E.append(np.amax(df.iloc[i][6:]) - np.amin(df.iloc[i][6:]))
        fit.append(linFit(df.iloc[i][6:]))
        lowA.append(lowCurr(df.iloc[i][6:]))
    AE = np.array(np.array(A)/np.array(E))
    plt.figure(0)
    for i in range(0, 150):
        if (AE[i] > -0.) & (E[i] > 300) & (fit[i][0] > -.3) & (fit[i][0] < -.15) & (fit[i][2] < -.4) & (lowA[i] > -35):
            print(A[i], lowA[i], AE[i], E[i], fit[i])
            plt.figure(0)
            plt.plot((df.iloc[i][6:] - np.mean(df.iloc[i][6:50])), alpha=0.8)
            plt.show()
            count += 1
    #plt.xlim(200,500)
    plt.xlabel("Time [arb]")
    plt.ylabel("Energy [arb]")

    plt.figure(1)
    plt.scatter(E, AE, s=4)
    plt.xlabel("Energy [arb]")
    plt.ylabel("A/E")
    print(count)
    plt.show()

   
def maxCurr(wf):
    current = np.gradient(wf)
    maximum = np.amax(current)
    minimum = np.amin(current)
    if np.abs(minimum) > maximum:
        return(minimum)
    else:
        return(maximum)

def lowCurr(wf):
    return np.min(np.gradient(wf))

def linFit(wf):
    x = np.linspace(0, 2048, 2048)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x[500:], wf[500:])
    return (slope, intercept, r_value)

if __name__ == '__main__':
    main(sys.argv[1])
