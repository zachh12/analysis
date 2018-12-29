#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np

import dnest4

from waffle.management import WaveformFitManager
from waffle.models import VelocityModel, LowPassFilterModel, HiPassFilterModel, ImpurityModelEnds
from waffle.models import FirstStageFilterModel, AntialiasingFilterModel
from waffle.models import OvershootFilterModel,OscillationFilterModel, TrappingModel
from siggen import PPC

chan_dict = {
600: "B8482",
626: "P42574A",
640:"P42665A",
648:"P42664A",
672: "P42661A",
692: "B8474"
}

def main(wf, doPlot=False):

    align_point = 0.95
    wf_idx = int(wf)

    chan = 626
    #chan = 692
    directory = "chan{}_wfs".format(chan)

    wf_file = "training_data/chan{}_2614wfs.npz".format(chan)
    conf_name = "{}.conf".format( chan_dict[chan] )

    datadir= os.environ['DATADIR']
    conf_file = datadir +"/siggen/config_files/" + conf_name


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


    data = np.load(wf_file, encoding="latin1")
    wfs = data['wfs']

    wf = wfs[wf_idx]
    wf_directory = os.path.join(directory, "wf{}".format(wf_idx))
    if os.path.isdir(wf_directory):
        if len(os.listdir(wf_directory)) >0:
            raise OSError("Directory {} already exists: not gonna over-write it".format(wf_directory))
    else:
        os.makedirs(wf_directory)

    wf.window_waveform(time_point=align_point, early_samples=100, num_samples=125)

    fm = WaveformFitManager(wf, align_percent=align_point, detector=detector, align_idx=100)

    fm.fit(numLevels=1000, directory = wf_directory, new_level_interval=1000, numParticles=3)


if __name__=="__main__":
    main(*sys.argv[1:])
