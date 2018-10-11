import pandas as pd
from siggen import PPC
from waffle.processing import *
import numpy as np
import dnest4.classic as dn4
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

chan_dict = {
600: "B8482",
626: "P42574A",
640:"P42665A",
648:"P42664A",
672: "P42661A",
692: "B8474"
}

chan = 626
avse = [0,6]

def main():
    #process(chan, avse)
    search()

def generateDataFrame(chan):
    cols = ['training_id', 'r', 'z', 'phi', 'ecal', 'avse', 'drift_time', 'drift_length']
    df = pd.DataFrame(columns=cols)
    name = "data/chan" + str(chan) + "data.h5"
    try:
        df.to_hdf(name, key='data')
    except:
        os.mkdir("data")
        df.to_hdf(name, key='data')

def getDataFrame():
    name = "data/chan" + str(chan) + "data.h5"
    try:
        df = pd.read_hdf(name, key='data')
    except:
        generateDataFrame(chan)
        df = pd.read_hdf(name, key='data')
    return df

def getWf(det, idx, r, z, theta):
    wfList = np.load("../Fit/training_data/chan626_5000wfs.npz")
    trainingIdx = wfList['wfs'][idx].training_set_index
    trainingSet = pd.read_hdf("../Fit/training_data/training_set.h5")
    #['training_id', 'r', 'z', 'phi', 'ecal', 'avse', 'drift_time', 'drift_length']
    wf = [trainingIdx, r, z, theta, trainingSet['ecal'][trainingIdx], 
        trainingSet['ae'][trainingIdx], trainingSet['drift_time'][trainingIdx], 
            getDriftLength(det, r, theta, z)]
    if (type(trainingSet['ecal'][trainingIdx]) != np.float64):
        return -1
    elif (trainingSet['ecal'][trainingIdx] < 2604):
        return -1
    else:
        return wf

def store(idx, r, theta, z):
    conf_name = "{}.conf".format(chan_dict[chan])
    datadir= os.environ['DATADIR']
    conf_file = datadir + "/siggen/config_files/" + conf_name
    det = PPC(conf_file, wf_padding=100)

    wfs = []
    for index in idx:
        wf = getWf(det, index, r[index], z[index], theta[index])
        if (wf == -1):
            continue;
        else:
            wfs.append(wf)

    df = getDataFrame()
    for wf in wfs:
        df.loc[len(df)] = wf

    plt.hist(df['ecal'])
    plt.show()
    exit()

def search():
    owd = os.getcwd()
    idx, r, theta, z = [], [], [], []
    for root, dirs, files in os.walk("../Fit/chan626_0-6avsewfs/"):
        for wf in dirs:
            chain = np.loadtxt(root + wf + "/posterior_sample.txt")
            rtemp, ztemp, thetatemp = [], [], []
            for sample in chain:
                try:
                    rtemp.append(sample[0])
                    ztemp.append(sample[1])
                    thetatemp.append(sample[2])
                except:
                    rtemp.append(chain[0])
                    ztemp.append(chain[1])
                    thetatemp.append(chain[2])
            idx.append(int(wf[2:]))
            r.append(rtemp[0])
            z.append(ztemp[0])
            theta.append(thetatemp[0])
    os.chdir(owd)
    store(idx, r, theta, z)

def process(chan, avse):
    owd = os.getcwd()
    name = "chan" + str(chan) + "_" + str(avse[0]) + "-" + str(avse[1]) + "avsewfs"
    os.chdir("../Fit/" + name)
    for root, dirs, files in os.walk("."):
        for wf in dirs:
            os.chdir(wf)
            dn4.postprocess(plot=False)
            os.chdir("..")
    os.chdir(owd)
def getDriftLength(det, r, theta, z):
    wf = det.GetWaveform(r, theta, z)
    hpath = det.siggenInst.GetPath(1)

    #r, hpath[0] ||| angle, hpath[1] ||| z, hpath[2]
    x = hpath[0]*np.cos(hpath[1])
    y = hpath[0]*np.sin(hpath[1])
    z = hpath[2]
    length = 0
    for k in range(1, len(x)-8):
        length += np.sqrt((x[k]-x[k-1])**2 + 
            (y[k]-y[k-1])**2 + (z[k]-z[k-1])**2)

    return length

if __name__ == "__main__":
    main()