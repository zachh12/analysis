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

    chan = 672
    #chan = 692
    directory = "chan{}_wfs".format(chan)

    #wf_file = "training_data/chan{}_2614wfs.npz".format(chan)
    wf_file = "training_data/datarun11510-11549chan672_250wfs.npz"
    conf_name = "{}.conf".format( chan_dict[chan] )

    datadir= os.environ['DATADIR']
    conf_file = datadir +"/siggen/config_files/" + conf_name


    detector = PPC( conf_file, wf_padding=100)

    vm = VelocityModel(include_beta=False)
    #hp1 = HiPassFilterModel(detector)
    #hp2 = HiPassFilterModel(detector)
    fs = FirstStageFilterModel(detector)
    al = AntialiasingFilterModel(detector)
    oshoot = OvershootFilterModel(detector)
    osc = OscillationFilterModel(detector)
    im = ImpurityModelEnds(detector)
    #tm = TrappingModel()

    #lp = LowPassFilterModel(detector)
    #hp = HiPassFilterModel(detector)

    det_params = [ 9.76373631e-01,8.35875049e-03,-5.09732644e+00,-6.00749043e+00,
                   4.74275220e+06,3.86911389e+06,6.22014783e+06,5.22077471e+06,
                    -3.63516477e+00,-4.48184667e-01]
    '''
    #OLD
    vm.apply_to_detector([8439025,5307015,9677126,5309391], detector)
    fs.apply_to_detector([-7.062660481537297308e-01, 9.778840228405032420e-01, -7.851819989636383390e+00], detector)
    al.apply_to_detector([0.5344254, 0.135507736], detector)
    oshoot.apply_to_detector([-3.8111099842, 0.140626600], detector)
    osc.apply_to_detector([-1.6800848, 3.00040593, -1.245777055, 5.0073780147], detector)
    im.apply_to_detector([-0.015571734108306538, -5.7326715464], detector)

    #672
    vm.apply_to_detector([6.330448119432486594e+06, 7.070545190569272265e+06, 6.330662440609236248e+06, 7.320939440024248324e+06], detector)
    fs.apply_to_detector([-1.50887, 9.790592e-01, -2.10503], detector)
    al.apply_to_detector([7.99097e-01, 1.160481e-02], detector)
    oshoot.apply_to_detector([-5.301815, 1.8299623], detector)
    osc.apply_to_detector([-2.185584, 6.970590, -2.2064522, 5.77401], detector)
    im.apply_to_detector([-2.739048e-01, -1.54175], detector)
    '''

    #try
    vm.apply_to_detector([6.329044e+06, 7.070545190569272265e+06, 6.3290662440e+06, 7.320139440024248324e+06], detector)
    fs.apply_to_detector([-1.51087, 9.790192e-01, -2.10403], detector)
    al.apply_to_detector([7.99097e-01, .0091], detector)
    oshoot.apply_to_detector([-5.2901815, 1.80], detector)
    osc.apply_to_detector([-2.181, 7, -2.2, 5.], detector)
    im.apply_to_detector([-.12, -1.54175], detector)
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
