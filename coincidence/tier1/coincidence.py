import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def main():
    run = 900
    df = pd.read_hdf("t1_run" + str(run) + ".h5", key="ORSIS3302DecoderForEnergy")


if __name__ == '__main__':
    main()