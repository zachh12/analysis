import dnest4 as dn4
import os
import sys
def main(idx):
    print(idx)
    for root, dirs, files in os.walk("."):
        for wf in dirs:
            index = "wf" + str(idx)
            if wf == index:
                os.chdir(wf)
                dn4.postprocess(plot=False)
                os.chdir("..")
        exit()

if __name__ == "__main__":
    main(*sys.argv[1:2])