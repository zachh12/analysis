import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4.classic as dn
from scipy import stats

chan_dict = {
600: "B8482",
626: "P42574A",
640:"P42665A",
648:"P42664A",
672: "P42661A",
692: "B8474"
}
def calibrate(list):

    df = np.load("../training_data/chan626_2614wfs.npz")
    df2 = pd.read_hdf("../training_data/training_set.h5")

    indexes = []
    times = []
    Energy = []
    for i in list:
        indexes.append((df['wfs'][i]).training_set_index)
        #dir = "../chan692_8wfs/wf" + str(int(i)) + "/posterior_sample.txt"
        #post = np.loadtxt(dir)
    for index in indexes:
        Energy.append(df2['ecal'][index])
        times.append(df2['drift_time'][index])

    return Energy, times

def getDriftLength(det, r, theta, z):
    wf = det.GetWaveform(r, theta, z)
    hpath = det.siggenInst.GetPath(1)

    #r, hpath[0] ||| angle, hpath[1] ||| z, hpath[2]
    x = hpath[0]*np.cos(hpath[1])
    y = hpath[0]*np.sin(hpath[1])
    z = hpath[2]
    length = 0
    for k in range(1, len(x)-8):
        #print(length)
        length += np.sqrt((x[k]-x[k-1])**2 + 
            (y[k]-y[k-1])**2 + (z[k]-z[k-1])**2)
    #print(length)
    #exit()
    return length


def energy(list, energy, times):

    chan = 626
    conf_name = "{}.conf".format( chan_dict[chan] )
    datadir= os.environ['DATADIR']
    conf_file = datadir +"/siggen/config_files/" + conf_name
    det = PPC( conf_file, wf_padding=100)
    r = []
    z = []
    theta = []
    efit = []
    drift_length = []
    for i in list:
        dir = "../chan626_2614wfs/wf" + str(int(i)) + "/posterior_sample.txt"
        post = np.loadtxt(dir)
        rtemp, ztemp, thetatemp, etemp = [], [], [], []
        for sample in post:
            try:
                rtemp.append(sample[0])
                ztemp.append(sample[1])
                thetatemp.append(sample[2])
                etemp.append(sample[3])
            except:
                rtemp.append(post[0])
                ztemp.append(post[1])
                thetatemp.append(post[2])
                etemp.append(post[3])
        r.append(rtemp[0])
        z.append(ztemp[0])
        theta.append(thetatemp[0])
        efit.append(etemp[0])
    for i in range(0, len(r)):
        length = getDriftLength(det, r[i], theta[i], z[i])
        drift_length.append(length)
    drift_length = np.array(drift_length)

    r = np.array(r)
    slope, intercept, r_value, p_value, std_err = stats.linregress(drift_length, energy)

    plt.figure(1)
    plt.title("Drift Length vs Drift Time (95%)")
    plt.scatter(times, drift_length)
    plt.ylabel("Drift Length (mm)")
    plt.xlabel("Drift Time (us)")

    plt.figure(2)
    plt.scatter(np.square(r), z)
    plt.xlabel("Radius^2 (mm)")
    plt.ylabel("Z (mm)")

    plt.figure(3)
    plt.title("Drift Length vs Energy @2614 keV Peak")
    plt.scatter(drift_length, energy)
    x = np.linspace(1, 300)
    y = x * slope + intercept
    regression = plt.plot(x, y, label="R: " + str(('%.5f'%(r_value))))
    plt.legend(loc='upper right')
    plt.xlim(.7*np.min(drift_length), 1.1*np.max(drift_length))
    plt.ylim(2610, 2620)
    plt.xlabel("Drift Length (mm)")
    plt.ylabel("Energy (keV)")

    '''
    #Used to calculate tau
    times = np.array(times)
    plt.close()
    plt.figure(100)
    rez = []
    #rez2 = []

    xs = np.linspace(10000, 50000, 10000)
    for i in xs:
        energyc = energy * np.exp((times)*(i/100000))
        #energyct = energy * np.exp((drift_length+times)*(i/1000000000))
        rez.append(np.std(energyc)*2.35)
        #rez2.append(np.std(energyct)*2.35)
    plt.scatter(xs, rez)
    plt.title("FWHM Minimization")
    plt.xlabel("Tau Parameter")
    plt.ylabel("FWHM @2614 keV")
    #plt.scatter(xs, rez2)
    print(rez[np.argmin(rez)], xs[np.argmin(rez)])
    #print(rez2[np.argmin(rez2)], xs[np.argmin(rez2)])
    energyc = energy * np.exp((drift_length)*(xs[np.argmin(rez)]/1000000000))
    #energyct = energy * np.exp((drift_length+times)*(xs[np.argmin(rez2)]/1000000000))
    plt.figure(20)
    plt.hist(energy, alpha=.5, bins=20)
    plt.hist(energyc, alpha=.5, bins=20)
    #plt.hist(energyct, alpha=.5, bins=20)
    #print(np.mean(energyc), np.mean(energyct))
    plt.show()
    exit()
    #'''
    tau = 5.14622924584917
    times = np.array(times)
    energyb = energy * np.exp((times)*(2.32213/100000))
    energyc = energy * np.exp((drift_length)*(tau/1000000))
    print("Resolution None: ", np.std(energy)*2.35)
    print("Resolution Time: ", np.std(energyb)*2.35)
    print("Resolution Length: ", np.std(energyc)*2.35)
    improv = (-100*(np.std(energyc)*2.35-np.std(energyb)*2.35)/(np.std(energyb)*2.35))
    print("% Improv t -> DL: ", improv)

    plt.figure(4)
    plt.hist(energy - np.mean(energy), bins=12, alpha=.3)
    plt.hist(energyb - np.mean(energyb), bins=12, alpha=.5)
    plt.hist(energyc - np.mean(energyc), bins=12, alpha=.5)
    plt.xlabel("Energy Distribution around Mean")
    plt.title("Energy With and w/o Correction")

    plt.figure(300)
    plt.scatter(drift_length, energyc)
    plt.title("Correction Drift Length vs Energy")
    plt.xlabel("Drift Length (mm)")
    plt.ylabel("Energy (keV)")
    fwhm = np.std(energyc) * 2.35
    plt.axhline(y=np.mean(energyc), alpha=.5, color='r')
    plt.axhspan(np.mean(energyc) - fwhm/2, np.mean(energyc) + fwhm/2, alpha=0.2, color='green')
    #for i in range(0, len(energyc)):
    #    if ((energy[i] > 2614) and (drift_length[i] > 200)):
    #        print(list[i])

    plt.show()

def main():
    exclude = [1, 16, 27]
    #Check these later
    tempexclude = [34, 45]
    wfs = np.arange(0, 60)
    valid = np.setdiff1d(wfs,exclude)
    valid = np.setdiff1d(valid, tempexclude)

    Energy, times = calibrate(valid)
    #plt.hist(Energy, bins=50)
    #plt.show()
    #for i in range(0, len(Energy)):
    #    if (Energy[i] < 2612):
    #        print(i)
    #exit()
    energy(valid, Energy, times)

if __name__ == "__main__":
    main()