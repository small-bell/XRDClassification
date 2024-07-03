from tqdm import tqdm

from autoXRD.spectrum_generation import peak_shifts, intensity_changes, peak_broadening, mixed
import pymatgen as mg
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool, Manager
from pymatgen.core import Structure
import random
import math
import os
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d


# from pyxtal import pyxtal

class SpectraGenerator(object):

    def __init__(self, reference_dir, filepath, num_spectra=5, max_texture=0.6, min_domain_size=1.0,
                 max_domain_size=100.0, max_strain=0.04, min_angle=5.0, max_angle=110.0):

        self.ref_dir = reference_dir
        self.filepath = filepath
        self.num_spectra = num_spectra
        self.max_texture = max_texture
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.max_strain = max_strain
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.calculator = xrd.XRDCalculator()

    def augment(self, phase_info):

        struc, filename, class_name = phase_info[0], phase_info[1], phase_info[2]
        patterns = []

        """
        pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))
        angles, intensities = pattern.x, pattern.y
        steps = np.linspace(self.min_angle, self.max_angle, 4501)
        signals = np.zeros(steps.shape[0]) 

        for i, ang in enumerate(angles): 
            idx = np.argmin(np.abs(ang-steps))
            signals[idx] = intensities[i]
        """
        for i in range(self.num_spectra):
            myfilepath = self.filepath + "/" + class_name + '/%s%s.x' % (filename, i)
            if os.path.exists(myfilepath):
                continue
            # try:
            #     struc.get_space_group_info()[1]
            # except:
            #     print(filename + ' has problem with get_space_group_info')
            #     continue
            patterns = peak_shifts.main(struc, self.max_strain, self.min_angle, self.max_angle, self.min_domain_size,
                                        self.max_domain_size)
            # patterns += peak_broadening.main(struc, self.num_spectra, self.min_domain_size, self.max_domain_size, self.min_angle, self.max_angle)
            # patterns += intensity_changes.main(struc, self.num_spectra, self.max_texture, self.min_angle, self.max_angle)
            # patterns += mixed.main(struc, self.num_spectra, self.max_strain, self.min_domain_size, self.max_domain_size,  self.max_texture, self.min_angle, self.max_angle)

            f = open(myfilepath, "x")
            for j in patterns:
                f.write(str(j) + '\n')
            f.close()
        print("{} finished".format(filename))
        return (patterns, filename)

    @property
    def augmented_spectra(self):


        for i in range(1, 231):  # 1到230
            if i > 1:
                break
            print("==============start {}===============".format(i))
            pool = multiprocessing.Pool(processes=1)
            phases = []
            folder_name = os.path.join(self.ref_dir, str(i))
            if os.path.isdir(folder_name):
                if not os.path.exists(os.path.join(self.filepath, str(i))):
                    os.makedirs(os.path.join(self.filepath, str(i)))
                for filename in os.listdir(folder_name):
                    if os.path.exists(self.filepath + "/" + str(i) + '/%s%s.x' % (filename, "0")):
                        continue
                    try:
                        phases.append(
                            [Structure.from_file('%s/%s/%s' % (self.ref_dir, str(i), filename)), filename, str(i)])
                    except:
                        pass
            pool.map(self.augment, phases)
            pool.close()
            pool.join()
            print("==============end {}===============".format(i))

        return "success"




