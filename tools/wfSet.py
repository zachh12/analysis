import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygama
from pygama.calculators import t0_estimate, max_time
from pygama.transforms import pz_correct
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
from scipy import stats
import os
import h5py

cal = [1173, 1332, 1460]
adc = [211, 246, 273]
def main():
    df = np.load("wfs.npz")
    #print(df.files)
    wfs0 = df['arr_0']
    wfs = []
    bl_ints, bl_stds, bl_slopes  = getBaselines(wfs0)

    for wf, bl_int, bl_std, bl_slope in zip(wfs0, bl_ints, bl_stds, bl_slopes):
        #cut_slope = (bl_slope > -.02) & (bl_slope < .02)
        #cut_std = bl_std < 5
        #cut = cut_slope & cut_std
        #if (cut):
        wf = wf - bl_int
        wfs.append(wf)
    wf = wfs[5]
    plt.plot(wf/np.amax(wf))

    plt.plot(pz_correct(wf/np.amax(wf), 200),alpha=.3)
    plt.plot(pz_correct(wf/np.amax(wf), 250),alpha=.3)
    plt.plot(pz_correct(wf/np.amax(wf), 300),alpha=.3)
    plt.plot(pz_correct(wf/np.amax(wf), 350), alpha=.3)
    plt.plot(pz_correct(wf/np.amax(wf), 400),alpha=.3)
    plt.plot(pz_correct(wf/np.amax(wf), 450),alpha=.3)
    plt.plot(pz_correct(wf/np.amax(wf), 470), alpha=.3)
    #plt.plot(pz_correct(wf/np.amax(wf), 75),alpha=.3)
    #plt.plot(pz_correct(wf/np.amax(wf), 80),alpha=.3)
        #plt.xlim(0, 500)
        #plt.ylim(0, 1)
    plt.show()
        #plt.plot(pz_correct(wfs[i]/np.amax(wfs[i]),180))
    plt.show()
    exit()
    trap_max = trap_maxes(wfs)

    slope, intercept, r_value, p_value, std_err = stats.linregress(adc, cal)
    trap_max = np.float64(trap_max) * slope + intercept 
    plt.hist(trap_max, bins=300)
    plt.show()

    np.savetxt("ecal.txt", trap_max)

def getBaselines(wfs):
    bl_int = []
    bl_std = []
    bl_slope = []

    x = np.linspace(1,250,250)
    for wf in wfs:
        bl = wf[0:250]
        bl_int.append(np.mean(bl))
        bl_std.append(np.std(bl))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, bl)
        bl_slope.append(slope)
    return bl_int, bl_std, bl_slope

def trap_maxes(wfs):
    trapMax = []

    for wf in wfs:
        wf = running_mean(wf, 3)
        trap = trap_filter(wf, rampTime=400, flatTime=200)
        #plt.plot(trap)
        #plt.plot(wf)
        #plt.ylim(0, np.amax(trap) + 50)
        #plt.show()
        trapMax.append(trap_max(trap, method="max", pickoff_sample=55))
        #trapMax.append(np.amax(wf))
    return trapMax

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def trap_filter(waveform, rampTime=200, flatTime=100, decayTime=0., baseline = 0.):
    """ Apply a trap filter to a waveform. """
    decayConstant = 0.
    norm = rampTime
    if decayTime != 0:
        decayConstant = 1./(np.exp(1./decayTime) - 1)
        norm *= decayConstant

    trapOutput = np.zeros_like(waveform)
    fVector = np.zeros_like(waveform)
    scratch = np.zeros_like(waveform)

    fVector[0] = waveform[0] - baseline
    trapOutput[0] = (decayConstant+1.)*(waveform[0] - baseline)

    wf_minus_ramp = np.zeros_like(waveform)
    wf_minus_ramp[:rampTime] = baseline
    wf_minus_ramp[rampTime:] = waveform[:len(waveform)-rampTime]

    wf_minus_ft_and_ramp = np.zeros_like(waveform)
    wf_minus_ft_and_ramp[:(flatTime+rampTime)] = baseline
    wf_minus_ft_and_ramp[(flatTime+rampTime):] = waveform[:len(waveform)-flatTime-rampTime]

    wf_minus_ft_and_2ramp = np.zeros_like(waveform)
    wf_minus_ft_and_2ramp[:(flatTime+2*rampTime)] = baseline
    wf_minus_ft_and_2ramp[(flatTime+2*rampTime):] = waveform[:len(waveform)-flatTime-2*rampTime]

    scratch = waveform - (wf_minus_ramp + wf_minus_ft_and_ramp + wf_minus_ft_and_2ramp )

    if decayConstant != 0:
        fVector = np.cumsum(fVector + scratch)
        trapOutput = np.cumsum(trapOutput +fVector+ decayConstant*scratch)
    else:
        trapOutput = np.cumsum(trapOutput + scratch)

    # Normalize and resize output
    trapOutput[:len(waveform) - (2*rampTime+flatTime)] = trapOutput[2*rampTime+flatTime:]/norm
    trapOutput.resize( (len(waveform) - (2*rampTime+flatTime)))
    return trapOutput

#Calculate maximum of trapezoid -- no pride here
def trap_max(waveform, method = "max", pickoff_sample = 0):
    if method == "max": return np.amax(waveform)
    elif method == "fixed_time": return waveform[pickoff_sample]

if __name__ == '__main__':
    main()