#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import matplotlib.pyplot as plt

from waffle.plots import WaveformFitPlotter
from waffle.models import VelocityModel, LowPassFilterModel, HiPassFilterModel, ImpurityModelEnds, WaveformModel
from waffle.models import FirstStageFilterModel, AntialiasingFilterModel
from waffle.models import OvershootFilterModel,OscillationFilterModel, TrappingModel

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

    vm = VelocityModel(include_beta=False)
    #lp = LowPassFilterModel(detector)
    #hp1 = HiPassFilterModel(detector)
    #hp2 = HiPassFilterModel(detector)
    fs = FirstStageFilterModel(detector)
    al = AntialiasingFilterModel(detector)
    oshoot = OvershootFilterModel(detector)
    osc = OscillationFilterModel(detector)
    im = ImpurityModelEnds(detector)
    tm = TrappingModel()

    #lp = LowPassFilterModel(detector)
    #hp = HiPassFilterModel(detector)

    det_params = [ 9.76373631e-01,8.35875049e-03,-5.09732644e+00,-6.00749043e+00,
                   4.74275220e+06,3.86911389e+06,6.22014783e+06,5.22077471e+06,
                    -3.63516477e+00,-4.48184667e-01]

    vm.apply_to_detector([8439025,5307015,9677126,5309391], detector)
    #lp.apply_to_detector([9.76373631e-01, 8.35875049e-03], detector)
    #hp1.apply_to_detector([-5.09732644e+00,-6.00749043e+00], detector)
    #hp2.apply_to_detector([1,1], detector)
    fs.apply_to_detector([-7.062660481537297308e-01, 9.778840228405032420e-01, -7.851819989636383390e+00], detector)
    al.apply_to_detector([0.5344254, 0.135507736], detector)
    oshoot.apply_to_detector([-3.8111099842, 0.140626600], detector)
    osc.apply_to_detector([-1.6800848, 3.00040593, -1.245777055, 5.0073780147], detector)
    im.apply_to_detector([-0.015571734108306538, -5.7326715464], detector)
    tm.apply_to_detector(2154.3, detector)
    wfm = WaveformModel(wf, align_percent=align_point, detector=detector, align_idx=100)
    plotter = WaveformFitPlotter(wf_directory, int(num_samples), wfm)

    plotter.plot_waveform()
    #plotter.plot_trace()

    plt.show()


if __name__=="__main__":
    main(*sys.argv[1:] )