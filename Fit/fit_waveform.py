#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np

import dnest4

from waffle.management import WaveformFitManager
from waffle.models import VelocityModel, LowPassFilterModel, HiPassFilterModel, ImpurityModelEnds
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
    directory = "chan{}_2614wfs".format(chan)

    wf_file = "training_data/chan{}_2614wfs.npz".format(chan)
    conf_name = "{}.conf".format( chan_dict[chan] )

    datadir= os.environ['DATADIR']
    conf_file = datadir +"/siggen/config_files/" + conf_name


    detector = PPC( conf_file, wf_padding=100)

    lp = LowPassFilterModel(detector)#, order=2)
    hp = HiPassFilterModel(detector)#, order=1)
    #hp2 = HiPassFilterModel(detector, order=1)
    im = ImpurityModelEnds(detector)#.imp_avg_lims, detector.imp_grad_lims, detector.detector_length)
    vm = VelocityModel(include_beta=False)

    #Zero
    det_params = [ 9.817430366089883176e-01, 1.278730138735418610e-02,7.199938218717272775e+01, -6.777932615671294236e+00,
                   6.449085470174515620e+06, 5.361424581672362983e+06, 6.976555702764152549e+06, 5.471965652032286860e+06,
                    -1.458587045254770842e-02, -7.898722765638677146e+00]
    #Free
    det_params = [9.730681131971946618e-01, 8.750163771125387541e-03,7.382672255928054028e+01, 3.298093005946304856e+03 ,
                   6.807978763477065600e+06, 5.345701859110661782e+06, 8.026928269081912003e+06, 5.366043547423805110e+06,
                    -1.550832342633457372e+00, 6.910839733938956009e-01]
    #672
    det_params = [9.730681131971946618e-01, 8.750163771125387541e-03,7.382672255928054028e+01, 3.298093005946304856e+03 ,
                   6.807978763477065600e+06, 5.345701859110661782e+06, 8.026928269081912003e+06, 5.366043547423805110e+06,
                    1.550832342633457372e+00, 6.910839733938956009e-01] 
    lp.apply_to_detector(det_params[:2], detector)
    hp.apply_to_detector(det_params[2:6], detector)
    vm.apply_to_detector(det_params[4:8], detector)
    im.apply_to_detector(det_params[8:], detector)

    '''lp.apply_to_detector([9.730681131971946618e-01, 8.750163771125387541e-03],detector)
    hp.apply_to_detector([7.382672255928054028e+01, 3.298093005946304856e+03,8.589024290551936502e-01, 1.330597419008175582e-02],detector)
    #hp2.apply_to_detector([8.589024290551936502e-01, 1.330597419008175582e-02],detector)
    im.apply_to_detector([-2.592151586040053468e-02, -1.496417924325905702e+00],detector)
    vm.apply_to_detector([6.807978763477065600e+06, 5.345701859110661782e+06,
         8.026928269081912003e+06, 5.366043547423805110e+06],detector)'''
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
    fm.fit(numLevels=1000, directory = wf_directory, new_level_interval=5000, numParticles=5)


if __name__=="__main__":
    main(*sys.argv[1:])
