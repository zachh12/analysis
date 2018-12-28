import dnest4 as dn4
import os, sys
import time
import numpy as np

def main(idx=0):
    print(idx)
    directory = "chan626_0-6avsewfs"
    os.chdir(directory)
    for root, dirs, files in os.walk("."):
        for wf in dirs:
            index = "wf" + str(idx)
            if wf == index:
                os.chdir(wf)
                loop = 1
                while loop == 1:
                    dn4.postprocess(plot=False)
                    post = np.loadtxt("posterior_sample.txt")
                    if (len(post) < 10):
                        time.sleep(150)
                    else:
                        os.chdir("../..")
                        return
            else:
                continue

if __name__ == "__main__":
    main(*sys.argv[1:2])
