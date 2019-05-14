import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4.classic as dn
from scipy import stats
from scipy.stats import norm
chan_dict = {
600: "B8482",
626: "P42574A",
640:"P42665A",
648:"P42664A",
672: "P42661A",
692: "B8474"
}
chan = 692


def main():
    plots()

def plots():

    df = getDataFrame()
    cut = (df['ecal'] > 2000)
    df = df[cut]

    conf_name = "{}.conf".format(chan_dict[chan])
    datadir= os.environ['DATADIR']
    conf_file = datadir + "/siggen/config_files/" + conf_name
    det = PPC(conf_file, wf_padding=100)
    t0 = []
    electron = []
    hole = []
    for i in range(0, len(df)):
        wf_e = np.copy(det.MakeWaveform(df['r'][i],df['phi'][i],df['z'][i],-1)[0])
        wf_h = np.copy(det.MakeWaveform(df['r'][i],df['phi'][i],df['z'][i],1)[0])
        hole.append(np.amax(wf_h))
        electron.append(np.amax(wf_e))
        t0.append(t0_estimate(wf_e + wf_h))
    hole = np.array(hole)
    electron = np.array(electron)
    t0 = np.array(t0)
    energy = df['trap_max']
  
    plt.figure(1)
    plt.scatter(energy, df['drift_time']/np.amax(df['drift_time']))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['drift_time'], energy)
    print(r_value)
    plt.figure(2)
    plt.scatter(energy, df['sim_hole_drift_time']/np.amax(df['sim_hole_drift_time']))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['sim_hole_drift_time'].astype('float'), energy)
    print(r_value)
    plt.show()
    #plt.hist(df['ecal'], histtype='step', lw=2,color='r', bins=15)

#Estimate t0
def t0_estimate(waveform, baseline=0, median_kernel_size=51, max_t0_adc=100):
    '''
    max t0 adc: maximum adc (above baseline) the wf can get to before assuming the wf has started
    '''

    #if np.amax(waveform)<max_t0_adc:
    #    return np.nan

    wf_med = signal.medfilt(waveform, kernel_size=median_kernel_size)
    med_diff = gaussian_filter1d(wf_med, sigma=1, order=1)

    tp05 = calc_timepoint(waveform, percentage=max_t0_adc, baseline=0, do_interp=False, doNorm=False)
    tp05_rel = np.int(tp05+1)
    thresh = 5E-5
    last_under = tp05_rel - np.argmax(med_diff[tp05_rel::-1]<=thresh)
    if last_under >= len(med_diff)-1:
        last_under = len(med_diff)-2

    t0 = np.interp(thresh, ( med_diff[last_under],   med_diff[last_under+1] ), (last_under, last_under+1))
    return t0

#Estimate arbitrary timepoint before max
def calc_timepoint(waveform, percentage=0.5, baseline=0, do_interp=False, doNorm=True, norm=None):
    '''
    percentage: if less than zero, will return timepoint on falling edge
    do_interp: linear linerpolation of the timepoint...
    '''
    wf_norm = (np.copy(waveform) - baseline)

    if doNorm:
        if norm is None: norm = np.amax(wf_norm)
        wf_norm /= norm

    def get_tp(perc):
        if perc > 0:
            first_over = np.argmax( wf_norm >= perc )
            if do_interp and first_over > 0:
                val = np.interp(perc, ( wf_norm[first_over-1],   wf_norm[first_over] ), (first_over-1, first_over))
            else: val = first_over
        else:
            perc = np.abs(perc)
            above_thresh = wf_norm >= perc
            last_over = len(wf_norm)-1 - np.argmax(above_thresh[::-1])
            if do_interp and last_over < len(wf_norm)-1:
                val = np.interp(perc, (  wf_norm[last_over+1], wf_norm[last_over] ), (last_over+1, last_over))
            else: val = last_over
        return val

    if not getattr(percentage, '__iter__', False):
        return get_tp(percentage)
    else:
        vfunc = np.vectorize(get_tp)
        return vfunc(percentage)
def getDataFrame():
    name = "data/chan692data.h5"
    #name = "data/chan"
    try:
        df = pd.read_hdf(name, key='data')
    except:
        print("Unable to read dataframe!")
        exit()
    return df

if __name__ == "__main__":
    main()