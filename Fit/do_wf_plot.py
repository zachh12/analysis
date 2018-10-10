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

    #lp = LowPassFilterModel(detector)
    #hp = HiPassFilterModel(detector, order=1)
    #im = ImpurityModelEnds(detector)
    #vm = VelocityModel(include_beta=False)
    lp = LowPassFilterModel(detector, order=2)
    hp = HiPassFilterModel(detector, order=1)
    hp2 = HiPassFilterModel(detector, order=1)
    im = ImpurityModelEnds(detector)#.imp_avg_lims, detector.imp_grad_lims, detector.detector_length)
    vm = VelocityModel(include_beta=False)
    det_params = [ 9.74075776165e-01,6.8539811564e-03,7.1578501727e+01, -5.438884396,
                   4.457515386644839868e+06,4.3184775276e+06,7.140984509584e+06,6.83450688e+06,
                    -8.9707028640156e-02,-1.630706781077]
    #672
    det_params = [9.730681131971946618e-01, 8.750163771125387541e-03,7.382672255928054028e+01, 3.298093005946304856e+03 ,
                   6.807978763477065600e+06, 5.345701859110661782e+06, 8.026928269081912003e+06, 5.366043547423805110e+06,
                    -1.550832342633457372e+00, 6.910839733938956009e-01]      
    #lp.apply_to_detector(det_params[:2], detector)
    #hp.apply_to_detector(det_params[2:4], detector)
    #vm.apply_to_detector(det_params[4:8], detector)
    #im.apply_to_detector(det_params[8:], detector)

    lp.apply_to_detector([9.730681131971946618e-01, 8.750163771125387541e-03],detector)
    hp.apply_to_detector([7.382672255928054028e+01, 3.298093005946304856e+03], detector)#,8.589024290551936502e-01, 1.330597419008175582e-02],detector)
    hp2.apply_to_detector([8.589024290551936502e-01, 1.330597419008175582e-02],detector)
    im.apply_to_detector([-2.592151586040053468e-02, -1.496417924325905702e+00],detector)
    vm.apply_to_detector([6.807978763477065600e+06, 5.345701859110661782e+06,
         8.026928269081912003e+06, 5.366043547423805110e+06],detector)
    wfm = WaveformModel(wf, align_percent=align_point, detector=detector, align_idx=100)
    plotter = WaveformFitPlotter(wf_directory, int(num_samples), wfm)

    plotter.plot_waveform()
    #plotter.plot_trace()

    plt.show()


if __name__=="__main__":
    main(*sys.argv[1:] )