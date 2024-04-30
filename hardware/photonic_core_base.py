# -*- coding: utf-8 -*-
# @Author: Hanqing Zhu
# @Date:   2023-02-07 11:02:13
# @Last Modified by:   Hanqing Zhu
# @Last Modified time: 2023-02-24 01:48:16
# photonic core base class

__all__ = ["PhotoniceCore"]

class PhotonicCore():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.photonic_core_type = None
        self.width = None
        self.height = None
        
    def cal_insertion_loss(self):
        raise NotImplementedError
    
    # calculate required parameters for energy prediction
    # power * work freq
    # input: dac + eo(tx)
    # output: oe(rx) + adc
    def cal_TX_energy(self):
        raise NotImplementedError
    
    def cal_D2A_energy(self):
        raise NotImplementedError
    
    def cal_RX_energy(self):
        raise NotImplementedError
    
    def cal_A2D_energy(self):
        raise NotImplementedError
    
    def cal_comp_energy(self):
        raise NotImplementedError
    
    def cal_laser_energy(self):
        raise NotImplementedError
    
    def cal_core_area(self):
        raise NotImplementedError

    def cal_core_power(self):
        raise NotImplementedError