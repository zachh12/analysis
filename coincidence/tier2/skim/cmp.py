import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


runs = [900, 901, 902, 903, 904]

def main():
    primary = []
    backing = []
    for run in runs:
        e1 = np.loadtxt("E1" + str(run) + ".txt")
        for e in e1:
            primary.append(e)
        e2 = np.loadtxt("E2" + str(run) + ".txt")
        for e in e2:
            backing.append(e)

    slope, intercept, r_value, p_value, std_err = stats.linregress([643, 1141], [511, 1274])
    primary = np.array(primary) * slope + intercept

    slope, intercept, r_value, p_value, std_err = stats.linregress([240, 577], [511, 1274])
    backing = np.array(backing) * slope + intercept
    plot(primary, backing)


def plot(primary, backing):
    plt.figure(0)
    plt.hist(primary, bins=20000, histtype='step')
    plt.yscale('log')
    plt.xlim(0, 2000)

    plt.figure(1)
    plt.hist(backing, bins=10000, histtype='step')
    plt.xlim(0, 2000)

    plt.figure(2)
    plt.hist(backing+primary, bins=15000, histtype='step')
    plt.xlim(0, 2000)

    plt.figure(3)
    plt.scatter(backing, primary, s=0.05)
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.xlabel("Energy [keV]")
    plt.ylabel("Energy [keV]")
    plt.show()
if __name__ == '__main__':
    main()
#e0 = np.loadtxt
