import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygama.calculators import t0_estimate, max_time
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
from scipy import stats
import os
import h5py

cal = [1173, 1332, 1460]
adc = [181, 210, 235]
def main():
    df = pd.read_hdf("t1_run3.h5", key='ORSIS3302DecoderForEnergy')
    #df =  df[(df.index < 30000)]
    df =  df[(df.index < 1000)]
    #exit()
    #print(df.columns)
    #exit()
    bl_ints, bl_stds, bl_slopes  = getBaselines(df)
    t0s = get_t0(df, bl_ints)
    wfs = []
    good_t0 = []
    for wf, bl_int, bl_std, bl_slope in zip(df['waveform'], bl_ints, bl_stds, bl_slopes):
        #cut_slope = (bl_slope > -.02) & (bl_slope < .02)
        #cut_t0 = (t0 > 250) & (t0 < 350)
        #cut_int = (bl_int > -760) & (bl_int < -737)
        #cut_int = (bl_int > -730) & (bl_int < -713)
        #cut_std = bl_std < 7
        #cut = cut_slope  & cut_std & cut_int
        #cut = cut_t0
        #if (cut):
        #good_t0.append(t0)
        wf = wf[0]
        wf = wf - bl_int
        wfs.append(wf)
    
    #plt.hist(df['energy_wf'])
    #plt.show()
    #exit()
    np.savez('wfs.npz', wfs)
    exit()
    trap_max = trap_maxes(wfs)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(adc, cal)
    #trap_max = np.float64(trap_max) * slope + intercept 

    wfsSave = []
    for wf, trap, bl_slope in zip(df['waveform'], trap_max, bl_slopes):
        cut = ((trap > 0))# & (trap < 1600))
        cut_slope = (bl_slope > -.03) & (bl_slope < .03)
        #cut_std = bl_std < 10
        cut = cut & cut_slope #& cut_std
        if (cut):
            wf = wf[0]
            wfsSave.append(wf)

    plt.hist(trap_max, bins=3200)
    #plt.xlabel("Energy [keV]")
    #plt.xlim(1400, 1550)
    plt.show()
    np.savetxt("ecal.txt", trap_max)

def get_t0(df, bl_ints):
    t0 = []
    #dt = []
    for wf, bl_int in zip(df['waveform'], bl_ints):
        wf = wf[0]
        wf = wf - bl_int
        est = t0_estimate(wf)
        t0.append(est)
        #dt.append(max_time(wf) - est)
    print("woo")
    return t0


def getBaselines(df):
    bl_int = []
    bl_std = []
    bl_slope = []

    x = np.linspace(1,250,250)
    for wf in df['waveform']:
        wf = wf[0]
        bl = wf[0:250]
        bl_int.append(np.mean(bl))
        bl_std.append(np.std(bl))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, bl)
        bl_slope.append(slope)
    #plt.hist(bl_int, bins=500)
    #plt.xlim(-900, -500)
    #plt.xlabel("Baseline Integer [adc]")
    #plt.show()
    #exit()
    return bl_int, bl_std, bl_slope

def trap_maxes(wfs):
    trapMax = []

    for wf in wfs:
        #250, 200, 20
        wf = running_mean(wf, 3)
        trap = trap_filter(wf, rampTime=400, flatTime=200)
        #plt.plot(wf)
        #plt.plot(trap)
        #plt.axvline(x=85, color='r')
        #plt.ylim(0, np.amax(wf) + 50)
        #plt.show()
        trapMax.append(trap_max(trap, method="fixed_time", pickoff_sample=55))
    return trapMax

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#def running_mean(x, N):
#    return np.convolve(x, np.ones((N,))/N, mode='valid')[(N-1):]

'''
    These are the helper functions used when
    processing data... These are just basic
    versions of those used within GAT or
    Pygama, however the outcome should
    be roughly the same
'''
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