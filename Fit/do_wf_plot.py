#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import matplotlib.pyplot as plt

from waffle.plots import WaveformFitPlotter
from waffle.models import VelocityModel, LowPassFilterModel, HiPassFilterModel, ImpurityModelEnds, WaveformModel

import numpy as np
from siggen import PPC
chan_dict = {
600: "B8482",
626: "P42574A", #All there
640:"P42665A",
648:"P42664A",
672: "P42661A", #Mostly good
692: "B8474" #Have this too
}

def main(dir_name, wf_idx, num_samples=10 ):
    wf_idx = int(wf_idx)

    align_point = 0.95
    chan = 626
    directory = "chan{}_2614wfs".format(chan)

    wf_directory = os.path.join(dir_name, "wf{}".format(wf_idx))

    wf_file = "training_data/chan{}_2614wfs.npz".format(chan)
    conf_name = "{}.conf".format( chan_dict[chan] )

    datadir= os.environ['DATADIR']
    conf_file = datadir +"/siggen/config_files/" + conf_name

    data = np.load(wf_file, encoding="latin1")
    wfs = data['wfs']


    wf = wfs[wf_idx]
    wf.window_waveform(time_point=0.95, early_samples=100, num_samples=125)
    detector = PPC( conf_file, wf_padding=100)

    lp = LowPassFilterModel(detector)
    hp = HiPassFilterModel(detector)
    im = ImpurityModelEnds(detector)
    vm = VelocityModel(include_beta=False)

    det_params = [ 9.76373631e-01,8.35875049e-03,-5.09732644e+00,-6.00749043e+00,
                   4.74275220e+06,3.86911389e+06,6.22014783e+06,5.22077471e+06,
                    -3.63516477e+00,-4.48184667e-01]

    lp.apply_to_detector(det_params[:2], detector)
    hp.apply_to_detector(det_params[2:4], detector)
    vm.apply_to_detector(det_params[4:8], detector)
    im.apply_to_detector(det_params[8:], detector)
    wfm = WaveformModel(wf, align_percent=align_point, detector=detector, align_idx=100)
    plotter = WaveformFitPlotter(wf_directory, int(num_samples), wfm)

    plotter.plot_waveform()
    #plotter.plot_trace()

    plt.show()


if __name__=="__main__":
    main(*sys.argv[1:] )