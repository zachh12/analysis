import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygama
from pygama.calculators import t0_estimate, max_time
from pygama.transforms import pz_correct, remove_baseline
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
from scipy import stats
import os
import h5py

cal = [1173, 1332, 1460]
adc = [211, 246, 273]
def main():
    #df = pd.read_hdf("chan626data.h5")
    df = pd.read_hdf("chan626data.h5")

    wfs = df['waveform']

    wfz = []
    for wf, bl_int, bl_slope in zip(df['waveform'],  df['bl_int'], df['bl_slope']):
        wf = remove_baseline(wf, bl_0=bl_int, bl_1=bl_slope)
        wfz.append(wf)
    wfs = wfz
    trap_max = trap_maxes(wfs)
    trap_ft = trap_fts(wfs)

    df['trap_max'] = trap_max
    df['trap_ft'] = trap_ft
    df['bl_int'] = df['bl_int']

    df.to_hdf("mjd_energy.h5", key='data')



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

def trap_fts(wfs):
    trapMax = []

    for wf in wfs:
        wf = pz_correct(wf, rc=79)
        trap = trap_filter(wf, rampTime=400, flatTime=200)
        trapMax.append(trap_max(trap, method="fixed_time", pickoff_sample=600))
    return trapMax

def trap_maxes(wfs):
    trapMax = []

    for wf in wfs:
        wf = pz_correct(wf, rc=79)
        trap = trap_filter(wf, rampTime=400, flatTime=200)
        #plt.plot(trap)
        #plt.show()
        trapMax.append(trap_max(trap, method="max", pickoff_sample=600))
    return trapMax

def trap_filter(waveform, rampTime=400, flatTime=200, decayTime=0., baseline = 0.):
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