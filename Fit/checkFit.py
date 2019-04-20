import sys
import time
import os
import numpy as np
import dnest4 as dn4

def main(idx=0):
    time.sleep(120)
    directory = "chan672_wfs"
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
                    if len(post) < 10:
                        time.sleep(120)
                    else:
                        os.chdir("../..")
                        return

if __name__ == "__main__":
    main(*sys.argv[1:2])
