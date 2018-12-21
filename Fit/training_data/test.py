import h5py
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_hdf('training_set.h5', key='data')
plt.hist(df['drift_time'] * 10)
plt.show()