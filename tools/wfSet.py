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
    #df = np.load("wfs.npz")
    #wfs0 = df['arr_0']
    df0 = pd.read_hdf("chan626data.h5")
    wfs = df0['waveform']
    wfs0 = wfs
    rc, tail = GetDecay(wfs0)
    #print(df.files)
    #wfs0 = df['arr_0']
    wfs = []
    bl_ints, bl_stds, bl_slopes  = getBaselines(wfs0)
    t0s = Get_t0(wfs0)
    #np.savetxt("t0s.txt", t0s)  

    #exit()
    #for wf, bl_int, bl_std, bl_slope in zip(wfs0, bl_ints, bl_stds, bl_slopes):
    #for wf, rc_d, bl_int, in zip(wfs0, rc, bl_ints):
    for wf,  bl_int, in zip(wfs0,  bl_ints):
        #cut_slope = (bl_slope > -.02) & (bl_slope < .02)
        #cut_std = bl_std < 5
        #cut = cut_slope & cut_std
        #cut = (rc_d > 1000)
        #if (cut):
        wf = wf - bl_int
        wfs.append(wf)

    #plt.plot(wfs[0])
    #plt.show()
    #exit()
    trap_max = trap_maxes(wfs)

    #cols = ['waveform', 'trap_max', 'bl_int', 'bl_slope', 'bl_std', 'rc_decay']
    df = pd.DataFrame()
    df['waveform'] = wfs0.tolist()
    df['trap_max'] = trap_max
    df['bl_int'] = bl_ints
    df['bl_slope'] = bl_slopes
    df['bl_std'] = bl_stds
    df['rc_decay'] = rc
    df['tail'] = tail
    df['t0'] = t0s
    df['drift_time'] = df0['drift_time']
    df.to_hdf("test_data.h5", key='data')

def Get_t0(wfs):
    t0s = []
    for wf in wfs:
        est = t0_estimate(wf)
        t0s.append(est)
    return t0s

def GetDecay(wfs):
    tail = []
    rc = []
    for wf in wfs:
        wf = wf / np.amax(wf)
        rc.append(FindRC(wf))
        wf = wf * 2614
        argmax = np.argmax(wf)
        wfTemp = wf
        wf = wf[argmax:]
        x = np.linspace(0, 1, len(wf))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, wf)
        tail.append(slope)
        #print(slope, intercept)
        #plt.plot(wfTemp)
        #plt.show()
    plt.figure(1)
    plt.hist(tail, bins=300)
    plt.xlabel("Decay Slope [ADC / Nanosecond???]")
    print(np.mean(tail))
    plt.figure(2)
    plt.hist(rc, bins=800)
    plt.xlabel("Optimized RC Constant [us]")
    #plt.xlim(-200, 100)
    #plt.yscale('log')
    plt.show()
    #exit()
    #plt.close()
    return rc, tail

def FindRC(wf):
    opt_rc = []
    min_slope = []
    pzs = np.linspace(-100, 500, 100)
    for pz in pzs:
        wf_pz = pz_correct(wf, rc=pz)
        argmax = np.argmax(wf_pz)
        wf_pz = wf_pz[argmax:]
        x = np.linspace(0, 1, len(wf_pz))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, wf_pz)
        opt_rc.append(pz)
        min_slope.append(slope)

    #plt.plot(wf)
    #plt.show()
    #print(opt_rc[np.argmin(np.abs(min_slope))])
    #plt.plot(pz_correct(wf, rc=opt_rc[np.argmin(np.abs(min_slope))]))
    #plt.show()
    return(opt_rc[np.argmin(np.abs(min_slope))])

def rc_decay(rc1_us, freq = 100E6):
    '''
    rc1_us: decay time constant in microseconds
    freq: digitization frequency of signal you wanna process
    '''

    rc1_dig= 1E-6 * (rc1_us) * freq
    rc1_exp = np.exp(-1./rc1_dig)
    num = [1,-1]
    den = [1, -rc1_exp]

    return (num, den)

def pz_correct(waveform, rc, digFreq=100E6):
    ''' RC params are in us'''
    #get the linear filter parameters.
    num, den = rc_decay(rc, digFreq)
    #reversing num and den does the inverse transform (ie, PZ corrects)
    return signal.lfilter(den, num, waveform)
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
        #wf = running_mean(wf, 3)
        trap = trap_filter(wf, rampTime=100, flatTime=200)
        #plt.plot(trap)
        #plt.plot(wf)
        #plt.ylim(0, np.amax(trap) + 50)
        #plt.xlim(0, 250)
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