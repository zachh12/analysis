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

chan = 672
setName = "."

def main():
    #process(chan)
    search()

def generateDataFrame(chan):
    cols = ['training_id', 'r', 'z', 'phi', 'ecal', 'avse', 'drift_time', 'hole_drift_length', 'electron_drift_length', 'sim_hole_drift_time', 
    'sim_electron_drift_time', 'fitE', 'waveform', 'bl_int', 'bl_slope']
    df = pd.DataFrame(columns=cols)
    name = "data/chan" + str(chan) + "data.h5"
    try:
        df.to_hdf(name, key='data')
    except:
        os.mkdir("data")
        df.to_hdf(name, key='data')

def getDataFrame():
    name = "data/chan" + str(chan) + "data.h5"
    #try:
    #    df = pd.read_hdf(name, key='data')
    #except:
    generateDataFrame(chan)
    df = pd.read_hdf(name, key='data')
    return df

def getWf(det, idx, r, z, theta, fitE):
    wfList = np.load("data/datarun11510-11549chan672_250wfs.npz")
    trainingIdx = wfList['wfs'][idx].training_set_index
    trainingSet = pd.read_hdf("data/datarun11510-11549.h5")
    #['training_id', 'r', 'z', 'phi', 'ecal', 'avse', 'drift_time', 'hole_drift_length', 'electron_drift_length']
    drift_lengths = getDriftLength(det, r, theta, z)

    wf = [trainingIdx, r, z, theta, trainingSet['ecal'][trainingIdx], 
        trainingSet['ae'][trainingIdx], trainingSet['drift_time'][trainingIdx], 
        drift_lengths[0], drift_lengths[1], drift_lengths[2], drift_lengths[3],
        fitE, trainingSet['waveform'][trainingIdx], wfList['wfs'][idx].bl_int, wfList['wfs'][idx].bl_slope]
    if (type(trainingSet['ecal'][trainingIdx]) != np.float64):
        return -1
    elif (trainingSet['ecal'][trainingIdx] < 500):
        return -1
    else:
        return wf

def store(idx, r, theta, z, tempE):

    conf_name = "{}.conf".format(chan_dict[chan])
    datadir= os.environ['DATADIR']
    conf_file = datadir + "/siggen/config_files/" + conf_name
    det = PPC(conf_file, wf_padding=100)
    df = getDataFrame()
    wfs = []
    for i in range(0, len(idx)):
        wf = getWf(det, idx[i], r[i], z[i], theta[i], tempE[i])
        if (wf == -1):
            continue;
        elif (df['training_id'].any() == wf[0]):
            print("Skip")
            continue;
        else:
            wfs.append(wf)

    for wf in wfs:
        df.loc[len(df)] = wf
    name = "data/chan" + str(chan) + "data.h5"
    df.to_hdf(name, key='data')

def search():
    owd = os.getcwd()
    idx, r, theta, z, fitE = [], [], [], [], []
    for root, dirs, files in os.walk("data/chan672_wfs/"):
        for wf in dirs:
            chain = np.loadtxt(root + wf + "/posterior_sample.txt")
            rtemp, ztemp, thetatemp, tempE = [], [], [], []
            for sample in chain:
                try:
                    rtemp.append(sample[0])
                    ztemp.append(sample[1])
                    thetatemp.append(sample[2])
                    tempE.append(sample[3])
                except:
                    rtemp.append(chain[0])
                    ztemp.append(chain[1])
                    thetatemp.append(chain[2])
                    tempE.append(chain[3])
            idx.append(int(wf[2:]))
            r.append(rtemp[0])
            z.append(ztemp[0])
            theta.append(thetatemp[0])
            fitE.append(tempE[0])

    os.chdir(owd)
    store(idx, r, theta, z, fitE)

def process(chan):
    owd = os.getcwd()
    name = setName
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

    x = hpath[0]
    y = hpath[1]
    z = hpath[2]

    hlength = 0
    for k in range(1, len(x)-8):
        hlength += np.sqrt((x[k]-x[k-1])**2 + 
            (y[k]-y[k-1])**2 + (z[k]-z[k-1])**2)
    epath = det.siggenInst.GetPath(0)

    x = epath[0]
    y = epath[1]
    z = epath[2]

    elength = 0
    for k in range(1, len(x)-8):
        elength += np.sqrt((x[k]-x[k-1])**2 + 
            (y[k]-y[k-1])**2 + (z[k]-z[k-1])**2)
    return hlength, elength, det.siggenInst.GetLastDriftTime(1), det.siggenInst.GetLastDriftTime(0)

if __name__ == "__main__":
    main()