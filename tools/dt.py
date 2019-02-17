import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygama import calculators


def main():
    df = pd.read_hdf("t1_run3.h5", key='ORSIS3302DecoderForEnergy')
    wf = df['waveform'][4][0]
    wf = wf + 750
    print(calculators.t0_estimate(wf))
    print(calculators.calc_timepoint(wf))
if __name__ == '__main__':
    main()