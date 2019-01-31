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
setName = "."
energies = []
def search():
    owd = os.getcwd()
    ene, idx, r, theta, z = [], [], [], [], []
    for root, dirs, files in os.walk("data/chan626_wfs/"):
        for wf in dirs:
            chain = np.loadtxt(root + wf + "/levels.txt")
            energies.append(chain[len(chain)-1][1])
    os.chdir(owd)
search()
plt.hist(energies)
plt.show()
wfList = np.load("data/chan626_250wfs.npz")







