import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygama
#from waffle.processing import *
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
from scipy import stats
import os
chans = [580, 626, 672, 692]
cal = [727.330, 860.564, 2614.553]
adc = [1784.87, 2117.04517, 6422.5466]
#580, 692
def main():
    #calibrate()
    wfs = np.load("data/datarun11510-11549chan626_250wfs.npz")
    df = pd.read_hdf("data/chan626data.h5", key='data')
    slope, intercept, r_value, p_value, std_err = stats.linregress(adc, cal)
    #print(slope, intercept, r_value)

#exit()

    fwhm = []
    #pzs = [40, 50, 60, 65, 68, 75, 89, 100, 120, 150]
    pzs = np.linspace(50, 100, 15)
    for pole in pzs:
        vals = []
        for i in range(250):
            wf = wfs['wfs'][i]
            if (wf.amplitude < 5000):
                continue

            blrm = remove_baseline(wf.get_waveform(), wf.bl_int, wf.bl_slope)
            pz = pz_correct(blrm, pole)
            trap = trap_filter(pz)
            vals.append(trap_max(trap, method="fixed_time", pickoff_sample=600))

        val = np.array(vals)
        ecal = val * slope + intercept
        print(np.std(ecal) * 2.35)
        fwhm.append(np.std(ecal) * 2.35)
    print(np.min(fwhm), pzs[np.argmin(fwhm)])
    pzs = np.array(pzs)
    plt.scatter(1/pzs, fwhm)
    plt.show()

    exit()


'''
Used to find ADC to ECAL conversion
'''
def calibrate(): 
    unFitwfs = np.load("data/datarun11510-11549chan626_250wfs.npz")
    #unFitTraining = pd.read_hdf("data/unfit/datarun11510-11550.h5", key="data")
    #print(dir(unFitwfs['wfs'][0]))
    adcs = []
    for i in range(60):
        wf = unFitwfs['wfs'][i]
        trap = trap_filter(pz_correct(remove_baseline(wf.get_waveform(), wf.bl_int, wf.bl_slope), 70))
        adcs.append(trap_max(trap,method="max"))
    one, two, tl = [], [], []
    for num in adcs:
        if (num > 5000):
            tl.append(num)
        elif (num < 1830):
            one.append(num)
        else:
            two.append(num)
    print(np.mean(one), np.mean(two), np.mean(tl))
    global adc
    adc = [np.mean(one), np.mean(two), np.mean(tl)]

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
