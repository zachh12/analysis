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
692: "B8474"
}

def main(dir_name, wf_idx, num_samples=20 ):
    wf_idx = int(wf_idx)

    align_point = 0.95
    chan = 692
    directory = "chan{}_8wfs".format(chan)

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
    det_params = [ 0.97349837791,0.0053163187,70.02657, -6.201,
                   5611159,5045252,7507524,6677169,
                    -3.63516477e+00,-4.48184667e-01]
    #Recent Fit
    det_params = [ 9.7605950989e-01,8.019304538e-03,7.05119327e+01, -5.479569,
                   5.36624076e+06,5.0848736e+06,6.98879195e+06,6.5106570258e+06,
                    -1.10748e-01,-8.16662213376e-02]
    det_params = [ 9.74075776165e-01,6.8539811564e-03,7.1578501727e+01, -5.438884396,
                   4.457515386644839868e+06,4.3184775276e+06,7.140984509584e+06,6.83450688e+06,
                    -8.9707028640156e-02,-1.630706781077]
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