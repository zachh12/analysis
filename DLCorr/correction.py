import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    df = getDataFrame()

    df = df[df['ecal'] > 2000]

    rp = np.sqrt(df['r']**2 + df['z']**2)
    fit_decay(df)

def fit_decay(df):
    check = np.linspace(-100, 700, 110)
    rez, rez2, rez3, rez4, rez5 = [], [], [], [], []
    sim_htime = np.array(df['sim_hole_drift_time'], dtype=float)
    sim_etime = np.array(df['sim_electron_drift_time'], dtype=float)
    dl_hole = np.array(df['hole_drift_length'], dtype=float)
    dl_electron = np.array(df['electron_drift_length'], dtype=float)
    rp = np.sqrt(98*dl_hole**2 + 2*dl_electron**2)
    for var in check:
        rez.append(np.std(df['trap_raw'] * np.exp(sim_htime*(var/10000000)))*2.35)
        rez2.append(np.std(df['trap_raw'] * np.exp(df['hole_drift_length'] * (var/10000000))) * 2.35)
        rez3.append(np.std(df['trap_raw'] * np.exp(df['drift_time'] * (var/10000000))) * 2.35)
        rez4.append(np.std(df['trap_raw'] * np.exp(rp * (var/10000000))) * 2.35)
    print("ECal: ", 2.35*np.std(df['ecal']))    
    print("Raw: ", 2.35*np.std(df['trap_raw']))
    print("FT: ", 2.35*np.std(df['trap_ft']))
    print("Max: ", 2.35*np.std(df['trap_max']))
    print("Hole Time: ", np.min(rez), check[np.argmin(rez)])
    print("Hole Length: ", np.min(rez2), check[np.argmin(rez2)])
    print("Drift Time: ", np.min(rez3), check[np.argmin(rez3)])
    print("RP: ", np.min(rez4), check[np.argmin(rez4)])
    check2 = np.linspace(-100, 400, 110)
    combs = []
    #for var in check:
    #    for var2 in check2:
    ##        rez4.append(np.std(df['trap_raw'] * np.exp(sim_htime*(var/10000000) + sim_etime*(var2/10000000)))*2.35)
    #        rez5.append(np.std(df['trap_raw'] * np.exp(dl_hole*(var/10000000) + dl_electron*(var2/10000000)))*2.35)
    #        combs.append((var, var2))
    #print(np.min(rez4), combs[np.argmin(rez4)])
    #print(np.min(rez5), combs[np.argmin(rez5)])

    #test = df['trap_raw'] * np.exp(dl_hole*(combs[np.argmin(rez5)][0]/10000000) + dl_electron*(combs[np.argmin(rez5)][1]/10000000))
    #plt.hist(test)
    #plt.show()
def getDataFrame():
    name = "data/trapC_chan626data.h5"
    try:
        df = pd.read_hdf(name, key='data')
    except:
        print("Unable to read dataframe!")
        exit()
    return df

if __name__ == "__main__":
    main()