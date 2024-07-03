import pymatgen as mg
import numpy as np
import random
import math
import os
from pymatgen.analysis.diffraction import xrd

import subprocess


class StrainGen(object):

    def __init__(self, struc, max_strain=0.04, min_angle=5.0, max_angle=110.0, min_domain_size=1, max_domain_size=100):
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.max_strain = max_strain
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.possible_domains = np.linspace(min_domain_size, max_domain_size, 100)

    @property
    def sg(self):
        return self.struc.get_space_group_info()[1]

    @property
    def conv_struc(self):
        sga = mg.symmetry.analyzer.SpacegroupAnalyzer(self.struc)
        return sga.get_conventional_standard_structure()

    @property
    def lattice(self):
        return self.struc.lattice

    @property
    def matrix(self):
        return self.struc.lattice.matrix

    @property
    def diag_range(self):
        max_strain = self.max_strain
        return np.linspace(1 - max_strain, 1 + max_strain, 1000)

    @property
    def off_diag_range(self):
        max_strain = self.max_strain
        return np.linspace(0 - max_strain, 0 + max_strain, 1000)

    @property
    def sg_class(self):
        sg = self.sg
        if sg in list(range(195, 231)):
            return 'cubic'
        elif sg in list(range(16, 76)):
            return 'orthorhombic'
        elif sg in list(range(3, 16)):
            return 'monoclinic'
        elif sg in list(range(1, 3)):
            return 'triclinic'
        elif sg in list(range(76, 195)):
            if sg in list(range(75, 83)) + list(range(143, 149)) + list(range(168, 175)):
                return 'low-sym hexagonal/tetragonal'
            else:
                return 'high-sym hexagonal/tetragonal'

    @property
    def strain_tensor(self):
        diag_range = self.diag_range
        off_diag_range = self.off_diag_range
        s11, s22, s33 = [random.choice(diag_range) for v in range(3)]
        s12, s13, s21, s23, s31, s32 = [random.choice(off_diag_range) for v in range(6)]
        sg_class = self.sg_class

        if sg_class in ['cubic', 'orthorhombic', 'monoclinic', 'high-sym hexagonal/tetragonal']:
            v1 = [s11, 0, 0]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v1 = [s11, s12, 0]
        elif sg_class == 'triclinic':
            v1 = [s11, s12, s13]

        if sg_class in ['cubic', 'high-sym hexagonal/tetragonal']:
            v2 = [0, s11, 0]
        elif sg_class == 'orthorhombic':
            v2 = [0, s22, 0]
        elif sg_class == 'monoclinic':
            v2 = [0, s22, s23]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v2 = [-s12, s22, 0]
        elif sg_class == 'triclinic':
            v2 = [s21, s22, s23]

        if sg_class == 'cubic':
            v3 = [0, 0, s11]
        elif sg_class == 'high-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'orthorhombic':
            v3 = [0, 0, s33]
        elif sg_class == 'monoclinic':
            v3 = [0, s23, s33]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'triclinic':
            v3 = [s31, s32, s33]

        return np.array([v1, v2, v3])

    @property
    def strained_matrix(self):
        return np.matmul(self.matrix, self.strain_tensor)

    @property
    def strained_lattice(self):
        return mg.core.Lattice(self.strained_matrix)

    @property
    def strained_struc(self):
        new_struc = self.struc.copy()
        new_struc.lattice = self.strained_lattice
        return new_struc

    @property
    def strained_spectrum(self):
        struc = self.strained_struc  # 结构对象
        pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))
        angles, intensities = pattern.x, pattern.y
        steps = np.linspace(self.min_angle, self.max_angle, 5250)
        signals = np.zeros(steps.shape[0])
        tau = random.choice(self.possible_domains)

        for i, ang in enumerate(angles):
            idx = np.argmin(np.abs(ang-steps))
            signals[idx] = intensities[i]

        conv = []
        for (ang, int) in zip(steps, signals): # 遍历x与y
            if int != 0: # 如果强度不为0
                # 基于Scherrer eqtn计算FWHM
                K = 0.9 # 形状系数
                wavelength = 0.15406 # Cu K-alpha在nm
                theta = math.radians(ang/2.) # 布拉格角(弧度)
                beta = (K / wavelength) * (math.cos(theta) / tau)

                # 将FWHM转换为std 偏差的高斯
                std_dev = beta/2.35482

                # 高斯函数的卷积
                gauss = [int*np.exp((-(val - ang)**2)/std_dev) for val in steps]
                conv.append(gauss) # 计入list：conv中

        mixed_data = zip(*conv) # 会将列表拆分成一个一个的独立元素，然后组合
        all_I = []
        for values in mixed_data: # 遍历数组
            noise = random.choice(np.linspace(-0.75, 0.75, 1000)) # 生成噪声随机值
            all_I.append(sum(values) + noise) # 加入噪声计入list：all_I

        shifted_vals = np.array(all_I) - min(all_I) # 偏移变量
        scaled_vals = 100*np.array(shifted_vals)/max(shifted_vals)
        all_I = [val for val in scaled_vals]
        
        """
        # 提取峰值
        conv = []
        for (ang, int) in zip(x, y):  # 遍历x与y
            if int != 0:  # 如果强度不为0
                gauss = [int * np.exp((-(val - ang) ** 2) / 0.15) for val in x]  # # 高斯函数的卷积
                conv.append(gauss)  # 计入list：conv中
        

        mixed_data = zip(*conv)  # 会将列表拆分成一个一个的独立元素，然后组合
        all_I = []
        for values in mixed_data:  # 遍历数组
            noise = random.choice(np.linspace(-0.75, 0.75, 1000))  # 生成噪声随机值
            all_I.append(sum(values) + noise)  # 加入噪声计入list：all_I

        shifted_vals = np.array(all_I) - min(all_I)  # 偏移变量
        scaled_vals = 100 * np.array(shifted_vals) / max(shifted_vals)
        all_I = [[val] for val in scaled_vals]
        """

        return all_I  # 返回偏移值


def main(struc , max_strain, min_angle=5.0, max_angle=110.0, min_domain_size=1, max_domain_size=100):  # 输入结构；默认:模拟每个相位的光谱数；默认:最大应变

    strain_generator = StrainGen(struc, max_strain, min_angle=5.0, max_angle=110.0, min_domain_size=1, max_domain_size=100)  # 将输入结构；默认:最大应变，带入类中处理

    strained_patterns = strain_generator.strained_spectrum  # 对每个光谱进行处理

    return strained_patterns  # 返回一个处理结果


