#!/usr/bin/env bash
numWaveforms=6
for ((i=0; i < $numWaveforms; ++i))
do
    python3 fit_waveform.py $i &
    ((++i))
    python3 fit_waveform.py $i &
    (sleep 10 && pkill -9 -f fit_waveform.py)
    #python3 process.py
done