import pandas as pd
from siggen import PPC
from waffle.processing import *
import numpy as np
import dnest4.classic as dn4
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

trap_dict = {
626: 80#64.28
}
chan = 626
def main():
    calcEnergy()
    calibrate()


def calibrate():
    cal = [727.330, 860.564, 2614.553]

    df = pd.read_hdf("data/chan" + str(chan) + "data.h5", key='data')

    df_max2614 = df[(df['trap_max'] > 6140) & (df['trap_max'] < 6230)]
    df_max2614mean = np.mean(df_max2614['trap_max'])
    df_max860 = df[(df['trap_max'] > 1800) & (df['trap_max'] < 1920)]
    df_max860mean = np.mean(df_max860['trap_max'])
    df_max727 = df[(df['trap_max'] > 1500) & (df['trap_max'] < 1600)]
    df_max727mean = np.mean(df_max727['trap_max'])
    adc = [df_max727mean, df_max860mean, df_max2614mean]

    slope, intercept, r_value, p_value, std_err = stats.linregress(adc, cal)
    df['trap_max'] = df['trap_max'] * slope + intercept

    df_max2614 = df[(df['trap_ft'] > 5050) & (df['trap_ft'] < 6230)]
    df_max2614mean = np.mean(df_max2614['trap_ft'])
    df_max860 = df[(df['trap_ft'] > 1600) & (df['trap_ft'] < 1850)]
    df_max860mean = np.mean(df_max860['trap_ft'])
    df_max727 = df[(df['trap_ft'] > 1200) & (df['trap_ft'] < 1500)]
    df_max727mean = np.mean(df_max727['trap_ft'])
    adc = [df_max727mean, df_max860mean, df_max2614mean]
    print(adc)
    exit()
    slope, intercept, r_value, p_value, std_err = stats.linregress(adc, cal)
    df['trap_ft'] = df['trap_ft'] * slope + intercept

    df_max2614 = df[(df['fitE'] > 6300) & (df['fitE'] < 6800)]
    df_max2614mean = np.mean(df_max2614['fitE'])
    df_max860 = df[(df['fitE'] > 2030) & (df['fitE'] < 2170)]
    df_max860mean = np.mean(df_max860['fitE'])
    df_max727 = df[(df['fitE'] > 1720) & (df['fitE'] < 1840)]
    df_max727mean = np.mean(df_max727['fitE'])
    adc = [df_max727mean, df_max860mean, df_max2614mean]
    print(adc)
    exit()
    slope, intercept, r_value, p_value, std_err = stats.linregress(adc, cal)
    df['fitE'] = df['fitE'] * slope + intercept

    print(np.std(df['trap_max']) * 2.35, np.std(df['trap_ft']) * 2.35, np.std(df['fitE']) * 2.35, np.std(df['ecal']) * 2.35)

    df.to_hdf("data/chan" + str(chan) + "data.h5", key='data')

def calcEnergy():
    df = pd.read_hdf("data/chan" + str(chan) + "data.h5", key='data')
    df['trap_ft'] = -1
    df['trap_max'] = -1

    for i in range(len(df)):
        wf = df['waveform'][i]
        blrm = remove_baseline(wf, df['bl_int'][i], df['bl_slope'][i])
        plt.plot(blrm)
        pz = pz_correct(wf, trap_dict[chan])
        trap = trap_filter(pz)
        #plt.plot(trap)
        #plt.show()
        #exit()
        df['trap_max'][i] = trap_max(trap, method="max")
        df['trap_ft'][i] = trap_max(trap, method="fixed_time", pickoff_sample=600)
    return df
    df.to_hdf("data/chan" + str(chan) + "data.h5", key='data')

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
