import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygama
#from waffle.processing import *
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
from scipy import stats
import os

def main():
    #df = np.load("wfs/datarun11510-11550chan626_50wfs.npz")
    #df = np.load("chan626_2614wfs.npz")
    wfs = np.load("data/datarun11510-11550chan626_500wfs.npz")
    df = pd.read_hdf("data/chan626data.h5", key='data')
    print(wfs['wfs'][0].bl_int)
    
    for i in range(30):
        print(dir(wfs['wfs'][0]))
        exit()
        wf = (((pz_correct(wfs['wfs'][i].get_waveform(), 89))))
        traps.append(trap_max(wf, method="fixed_time", pickoff_sample=1200))
        #plt.show()
        #wf = (pz_correct(wfs['wfs'][i].get_waveform(), 89))
        #wf = trap_filter(wf)
        #traps.append(trap_max(wf))


    #adc = [2100, np.mean(traps)]
    #cal = [860.564, 2614.553]
    #slope, intercept, r_value, p_value, std_err = stats.linregress(adc, cal)

    traps = np.array(traps)
    ecal = traps
    #ecal = (traps * (1/82))
    ecal = ecal + (2614.5 - np.mean(ecal))

    print(np.std(ecal)*2.35)
    plt.hist(ecal)
    plt.show()


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

#Finds average baseline from first [samples] number of samples
def remove_baseline(waveform, bl_0=0, bl_1=0):
    return (waveform - (bl_0 + bl_1*np.arange(len(waveform))))

if __name__ == '__main__':
    main()
