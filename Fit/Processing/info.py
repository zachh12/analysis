import pandas as pd
from siggen import PPC
from waffle.processing import *
import numpy as np
import dnest4.classic as dn
from scipy import stats
import os

chan_dict = {
600: "B8482",
626: "P42574A",
640:"P42665A",
648:"P42664A",
672: "P42661A",
692: "B8474"
}

def genDF(chan):
    cols = ['chan', 'training_id', 'avse', 'ecal', 'drift_length', 'drift_time', 
        'r', 'z', 'phi', ]
    df = pd.DataFrame(columns=cols)

def store(df, wfinfo):

def search(df, chan, avse):
    for root, dirs, files in os.walk("."):
        for wf in dirs:
            os.chdir(wf)
            chain = np.loadtxt(wf)
            wfinfo = [chain[0], chain[1], chain[2]]
            store(wfinfo)
            os.chdir("..")

def process():
    for root, dirs, files in os.walk("."):
        for wf in dirs:
            os.chdir(wf)
            dn4.postprocess(plot=False)
            os.chdir("..")

def getDriftLength():
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
def main():

if __name__ == "__main__":
    main()