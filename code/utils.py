import numpy as np
import pandas as pd
import os
import csv
from tsfresh import extract_features

def fft_ft(dt):
    
    from tsfresh.feature_extraction.feature_calculators import fft_coefficient
    
    params = []
    for i in range(10):
        for j in ['real', 'imag', 'abs', 'angle']:
            params.append({'coeff': i, 'attr': j})

    ft = fft_coefficient(dt, params)
    
    return {i[0]: i[1] for i in ft}


def feature_extract(dt):
    
    import tsfresh.feature_extraction.feature_calculators as fc
    
    ft = {
        'abs_energy': fc.abs_energy(dt),
        'sum_values': fc.sum_values(dt),
        'mean': fc.mean(dt),
        'maximum': fc.maximum(dt),
        'minimum': fc.minimum(dt),
        'median': fc.median(dt),
        'quantile_0.1': fc.quantile(dt, 0.1),
        'quantile_0.2': fc.quantile(dt, 0.2),
        'quantile_0.3': fc.quantile(dt, 0.3),
        'quantile_0.4': fc.quantile(dt, 0.4),
        'quantile_0.5': fc.quantile(dt, 0.5),
        'quantile_0.6': fc.quantile(dt, 0.6),
        'quantile_0.7': fc.quantile(dt, 0.7),
        'quantile_0.8': fc.quantile(dt, 0.8),
        'quantile_0.9': fc.quantile(dt, 0.9),
        #
        # TODO:
        # Below functions dont works well -> need to be checked!!
        #
        #'fft_coefficient__coeff_0__attr_real': fc.fft_coefficient(dt {"coeff": 0, "attr": "real"}),
        #'fft_coefficient__coeff_0__attr_imag': fc.fft_coefficient(dt {"coeff": 0, "attr": "imag"}),
        #'fft_coefficient__coeff_0__attr_abs': fc.fft_coefficient(dt {"coeff": 0, "attr": "abs"}),
        #'fft_coefficient__coeff_0__attr_angle': fc.fft_coefficient(dt {"coeff": 0, "attr": "angle"}),
        #
        #=> Mr. Huy just fix this issue with above function fft_ft !!
    }
    
    ft.update(fft_ft(dt))
    
    return ft