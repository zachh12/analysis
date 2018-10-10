import dnest4 as dn4
import os
for root, dirs, files in os.walk("."):
    for wf in dirs:
        os.chdir(wf)
        dn4.postprocess(plot=False)
        os.chdir("..")
    exit()

