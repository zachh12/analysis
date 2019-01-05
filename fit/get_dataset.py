#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *
import matplotlib.pyplot as plt
def main():

    runList = np.arange(11510, 11520)
    #Tier 1 must match Tier 2 Channels
    chanList = [580, 626, 672, 692]

    proc = DataProcessor(detectorChanList=chanList)

    #Pygama processing
    #runList = np.arange(11537, 11551)
    #proc.tier0(runList, chanList)
    #df = proc.tier1(runList, num_threads=4, overwrite=True)
    #exit()

    #Load all runs into one common DF
    df = proc.load_t2(runList)

    df = proc.tag_pulsers(df)
    df = df.groupby("channel").apply(proc.calibrate)
    df = df.groupby(["runNumber","channel"]).apply(proc.calculate_previous_event_params, baseline_meas="bl_int")

    #proc.calc_baseline_cuts(df, settle_time=25) #ms
    #proc.fit_pz(df)
    #proc.calc_ae_cut(df )

    #calculate cut of good training waveforms
    df_bl = pd.read_hdf(proc.channel_info_file_name, key="baseline")
    df_ae = pd.read_hdf(proc.channel_info_file_name, key="ae")
    df = df.groupby("channel").apply(proc.tag_training_candidates, df_bl=df_bl,df_ae=df_ae)

    proc.save_t2(df)

    proc.save_dataset(runList, "datasets/datarun{}-{}".format(np.min(runList), np.max(runList)))
    #exit(5)'''
    n_waveforms = 50
    for chan in chanList:
        proc.save_wfs(chan, n_waveforms, "datasets/datarun{}-{}".format(np.min(runList), np.max(runList)),
            "datasets/wfs/datarun{}-{}chan{}_{}wfs.npz".format(np.min(runList), np.max(runList), chan, n_waveforms))


if __name__=="__main__":
    main()