import pandas as pd
from siggen import PPC
from waffle.processing import *
import numpy as np
import dnest4.classic as dn4
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

def main():
    '''r1,r2,r3,r4,z1,z2,z3,z4 = [],[],[],[],[],[],[],[]
    sample = np.loadtxt("Posteriors/1.txt")
    for point in sample:
        r1.append(point[0])
        z1.append(point[1])
    sample = np.loadtxt("Posteriors/2.txt")
    for point in sample:
        r2.append(point[0])
        z2.append(point[1])
    sample = np.loadtxt("Posteriors/3.txt")
    for point in sample:
        r3.append(point[0])
        z3.append(point[1])
    sample = np.loadtxt("Posteriors/4.txt")
    for point in sample:
        r4.append(point[0])
        z4.append(point[1])
    #plt.scatter(r1,z1)
    #plt.scatter(r2,z2)
    #plt.xlim(0,40)
    #plt.ylim(0,40)
    #plt.show()

    '''
    #Calculate drift times
    # plt.figure()
    path = os.getenv("DATADIR")
    path = os.path.join(path,'siggen/config_files/P42661A.conf')
    det = PPC(path)
    for i in range(0, 500):
        wf = det.GetWaveform(4, 0, 20)
    exit(1)
    dt_list = []
    nr,nz=100,100
    dt_map = np.ones((nr,nz))*np.nan
    dtfrac_map = np.ones((nr,nz))*np.nan
    for ir,r in enumerate(np.linspace(0, det.detector_radius,nr)):
        for iz,z in enumerate(np.linspace(0, det.detector_length,nz)):
            wf = det.MakeWaveform(r,0,z, 1)
            if wf is None: continue
            dt_frac = np.argmax(wf >= 0.001) #pulls first index >= 0.1%
            dt = det.siggenInst.GetLastDriftTime(1)
            dt_map[ir,iz] = dt
            dtfrac_map[ir,iz] = dt_frac
            dt_list.append( dt  )
    color_map = plt.cm.RdYlBu_r
    normal_size = (7,8) #figure width x height, inches
    dt_arr = np.array(dt_list)
    plt.figure(figsize=normal_size)
    extent=(0, det.detector_radius, 0, det.detector_length)
    plt.imshow(dt_map.T, origin='lower', extent=extent, interpolation='nearest', cmap=color_map)
    plt.xlabel("Radial position [mm]")
    plt.ylabel("Axial position [mm]")
    levels = np.arange(200,2000,200)
    cb = plt.colorbar()
    cb.set_label('Drift Time [ns]', rotation=270, labelpad=10)
    CS = plt.contour(dt_map.T, levels, colors='k', origin='lower', extent=extent)
    plt.clabel(CS, inline=1, fmt='%1.0f ns', fontsize=10, inline_spacing=15)

    plt.tight_layout()
    plt.scatter(r1,z1,s=10)
    plt.scatter(r2,z2,s=10)
    plt.scatter(r3,z3,s=10)
    plt.scatter(r4,z4,s=10)
    plt.show()
if __name__ == "__main__":
    main()