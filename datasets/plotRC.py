import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygama
from waffle.processing import *
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
import os

def main():
    #df = np.load("datarun11510-11549chan672_250wfs.npz")
    df = np.load("chan626_2614wfs.npz")

    #df = np.load("../Fit/training_data/chan626_8wfs.npz")
    #df2 = pd.read_hdf(os.getenv("DATADIR") + "/mjd/t1/t1_run11510.h5", key="ORGretina4MWaveformDecoder")
    for i in range(150):
        plt.plot(df['wfs'][i].get_waveform())
    plt.show()


    #plt.plot(pz_correct(df2['waveform'][1355], 72))



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

if __name__ == '__main__':
    main()
