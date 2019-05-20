from siggen import PPC
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft

def main():

    det = PPC("config_files/p1_new.config")
    wf_h = np.copy(det.MakeWaveform(15,0,15,1)[0])
    wf_e = np.copy(det.MakeWaveform(15,0,15,-1)[0])
    #plt.plot(wf_h)
    #plt.plot(wf_e)
    #plt.plot(wf_h+wf_e)
    #plt.xlim(0, 800)
    #plt.show()
    #exit()
    wf = wf_h + wf_e
    dx = 1
    wf_ = np.gradient(wf, dx)
    #wf_ = np.gradient(wf_, dx)
    plt.xlim(0,600)
    plt.plot(wf_)
    plt.show()
    wf_h = np.gradient(wf_h, dx)
    wf_h = np.gradient(wf_h, dx)
    plt.plot(wf_h - wf_)
    plt.xlim(0, 500)
    plt.show()

if __name__ == '__main__':
    main()