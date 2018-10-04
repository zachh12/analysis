import numpy as np
import dnest4 as dn4
def main():
    folder = "../chan692_50wfs/wf"
    for i in range(0, 1):
        file = folder + str(i) +"/"
        dn4.postprocess(file, plot=False)

if __name__=="__main__":
    main()   