import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC
import dnest4.classic as dn
from scipy import stats

#Radius^2 1150mm
def main():
    list = np.arange(0, 21)
    r = []
    z = []
    theta = []
    drift_length = []
    for i in list:
        dir = "../chan626_2614wfs/wf" + str(int(i)) + "/posterior_sample.txt"
        post = np.loadtxt(dir)
        rtemp, ztemp, thetatemp = [], [], []
        for sample in post:
            try:
                rtemp.append(sample[0])
                ztemp.append(sample[1])
                thetatemp.append(sample[2])
            except:
                rtemp.append(post[0])
                ztemp.append(post[1])
                thetatemp.append(post[2])
        r.append(rtemp[0])
        z.append(ztemp[0])
        theta.append(thetatemp[0])
    r = np.array(r)
    plt.scatter(np.square(r), z)
    plt.show()

if __name__ == "__main__":
    main()