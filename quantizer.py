import numpy as np
import torch


_scale = 23.4838
_min = -23.1728

_max=20.3891

def quant_fix(features):
    for name, pyramid in features.items():
        pyramid_q = (pyramid-_min) * _scale
        features[name] = pyramid_q
    return features
    
def dequant_fix(x):
    return x.type(torch.float32)/_scale + _min
# -----------------------------------------------------------
# #MSFC——c17_6.1_14.5
# _scale_dsslic = 300.2181
# _min_dsslic = -1.7556

# #MSFC——D7
# _min_dsslic = -1.2516
# _scale_dsslic = 403.8514


#MSFC——C13
# _scale_dsslic = 258.0146
# _min_dsslic = -1.9635

#MSFC——C8
# _scale_dsslic = 312.6046
# _min_dsslic =-1.6080

def quant_fix_dsslic(features,_min_dsslic,_scale_dsslic):
    features_q = (features - _min_dsslic) * _scale_dsslic
    return features_q

def dequant_fix_dsslic(x,_min_dsslic,_scale_dsslic):
    return x.type(torch.float32) / _scale_dsslic + _min_dsslic

# -----------------------------------------------------------
#mask
_scale_mask = 1023
_min_mask = 0

def quant_fix_mask(features):
    features_q = (features - _min_mask) * _scale_mask
    return features_q

def dequant_fix_mask(x):
    return x.type(torch.float32) / _scale_mask + _min_mask



