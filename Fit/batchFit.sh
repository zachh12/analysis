#!/usr/bin/env bash
max=6
for ((i=0; i <= $max; ++i))
do
    python3 fit_waveform.py $i &
    #python3 fit_waveform.py 1 &
    (sleep 10 && pkill -9 -f fit_waveform.py)
    sleep 1
    killall python3
    echo "$i"
done