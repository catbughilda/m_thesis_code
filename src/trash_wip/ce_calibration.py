#%%
"CE MODEL CALIBRATION"
import pandas as pd
import numpy as np
import math
from scipy import optimize
from matplotlib import pyplot as plt
from src.helper import MAPE
# REMEMBER TO SET BOUNDARIES:
# a, b > 0
# sigma_x > 0
#%%

def CE_objective_fct(params, mkt_spread, maturities, params_cir):
    params_gbm, params_ce = params 
    k, mu, r0, sigma_r = params_cir
    a, b, c = params_ce
    x_ratio, alpha, sigma_x = params_gbm
    if sigma_r <= 0 or  sigma_x <= 0:
        return np.inf
    else:
        mod_spread = CDS_price_CE(0.4, maturities, params_cir, params_ce, params_gbm)
    
    return MAPE(mod_spread, mkt_spread) 


def glob_CE_calibration(mkt_spread, maturities, params_cir):
    bounds = [(0.00001, 1), 
                       (0.00001, 1), 
                       (0.00001, 0.1), 
                       (0.00001, 1),
                       (0.00001, 1),
                       (0.00001, 1)]

    # Global optimization
    # TOO MANY EVALUATIONS, CANT USE BRUTE WITH 6 PARAMS
    #res1 = optimize.brute(
    
    
# %%
#%%% Example

