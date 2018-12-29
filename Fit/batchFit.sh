#!/bin/bash
numWaveforms=450
for ((i=0; i < $numWaveforms; ++i))
do
    python3 fit_waveform.py $i &
    ((++i))
    python3 fit_waveform.py $i &
    k=$i
    ((--k))
    python3 checkFit.py $k &&
    python3 checkFit.py $i &&
    pkill -9 -f fit_waveform.py
done