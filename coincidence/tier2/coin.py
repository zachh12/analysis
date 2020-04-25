import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
def main(run):
    #channel 6 == coincidence
    df = pd.read_hdf("t2_run" + str(run) + ".h5", key='data')
    df = df.drop(['energy', 'energy_first', 'ts_hi', 'bl_rms', 'bl_p0', 'tp5', 'tp90', 'tp100', 'blsub_imax', 'bl_p1', 'current_imax'], axis=1)


    df6 = df[df['channel'] == 6]
    df6 = df6.reset_index(drop=True)
    df7 = df[df['channel'] == 7]
    df7 = df7.reset_index(drop=True)
    E1, E2 = [], []

    df6['ts'] = ts(df6)
    df7['ts'] = ts(df7)
    diff = []
    #df6 = df6[df6['ts'] < 200]
    #df7 = df7[df7['ts'] < 200]
    E1 = df6['blsub_max']
    for index, row in df6.iterrows():
        temp = df7[df7['ts'] > row['ts']]
        temp = temp.reset_index(drop=True)
        diff.append(row['ts'] - temp.iloc[0]['ts'])
        E2.append(temp.iloc[0]['blsub_max'])
    print(len(E1), len(E2))
    plt.scatter(E1, E2, s=1, alpha=0.3)
    plt.xlim(0, 2500)
    plt.ylim(0, 2500)
    np.savetxt("skim/E1" + str(run) + ".txt", E1)
    np.savetxt("skim/E2" + str(run) + ".txt", E2)
    np.savetxt("skim/diff" + str(run) + ".txt", diff)
    plt.show()

def ts(df):
    clock = 1e8
    UINT_MAX = 4294967295 # (0xffffffff)
    t_max = UINT_MAX / clock

    ts = df["timestamp"].values / clock
    tdiff = np.diff(ts)
    tdiff = np.insert(tdiff, 0 , 0)
    entry = np.arange(0, len(ts), 1)
    iwrap = np.where(tdiff < 0)
    iloop = np.append(iwrap[0], len(ts))

    ts_new, t_roll = [], 0

    for i, idx in enumerate(iloop):

        ilo = 0 if i==0 else iwrap[0][i-1]
        ihi = idx

        ts_block = ts[ilo:ihi]
        t_last = ts[ilo-1]
        t_diff = t_max - t_last
        ts_new.append(ts_block + t_roll)

        t_roll += t_last + t_diff # increment for the next block

    ts_wrapped = np.concatenate(ts_new)
    return ts_wrapped
if __name__ == '__main__':
    main(sys.argv[1])

