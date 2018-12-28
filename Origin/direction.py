import numpy as np
import matplotlib.pyplot as plt
from siggen import PPC
import os

def main():
    sample = np.loadtxt("sample.txt")
    post = np.loadtxt("Posteriors/posterior_sample.txt")
    r = np.array([])
    z = np.array([])
    for i in range(0, 400):
        r = np.append(r, sample[i][0])
        z = np.append(z, sample[i][1])
    pr = np.array([])
    pz = np.array([])
    for i in range(0, 10):
        pr = np.append(pr, post[i][0])
        pz = np.append(pz, post[i][1])
    plt.scatter(r, z, s=2)
    plt.scatter(pr, pz, s=3)
    plt.xlim(15, 25)
    plt.ylim(10, 25)
    plt.show()

if __name__ == "__main__":
    main()