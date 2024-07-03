import sys
import os

sys.path.append('./autoXRD')
from autoXRD import spectrum_generation

if __name__ == '__main__':

    max_texture = 0.5
    min_domain_size, max_domain_size = 5.0, 30.0
    max_strain = 0.03
    num_spectra = 3
    min_angle, max_angle = 5.0, 110.0

    orgpath = os.path.abspath(sys.path[0])

    xrd_obj = spectrum_generation.SpectraGenerator("./src/", "./dest/",
                                                   num_spectra, max_texture, min_domain_size, max_domain_size,
                                                   max_strain, min_angle, max_angle)
    xrd_specs = xrd_obj.augmented_spectra
    print(xrd_specs)