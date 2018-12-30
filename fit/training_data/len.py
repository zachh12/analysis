import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ts = pd.read_hdf("training_set.h5", key="data")
cut = (ts["channel"] == 626)
ts = ts[cut]
print(len(ts))
plt.hist(ts["ecal"])
plt.show()

