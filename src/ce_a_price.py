"CE MODEL PRICE"
#%%% CDS pricing under the CE model

''' Parameters that need to be estimated:
    - a, b, c=0 for the intensity rate, 
    - x_ratio, alpha, sigma_x for the GBM (credit quality variable) 
    GIVEN:
    - k, mu, r0, sigma_r from the CIR model (interest rate)
'''

from math import tau
import os
from dotenv import load_dotenv

load_dotenv()
path = os.getenv('macbook_path')
os.chdir(path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.cir_calibration import bond_price_CIR
from src.num_routines_odes import solve_CD
import scipy.integrate as integrate
from scipy.special import ndtr
from scipy import optimize
from src.helper import MAPE
import time


# The computations suppose a notional of 1, t_0 = 0, and continuous compounding 
#%% 
def find_q(params_ce, params_cir): #params_ce are the parameters of the intensity rate
    a, b, c = params_ce # intensity rate parameters, c unused here
    k, mu, r0, sigma_r = params_cir # CIR parameters from previous optimization

    q = np.sqrt(k**2 + 2*(b+1)*sigma_r**2) # b is unknown, k and sigma_r from CIR calibration, similar to q in CIR model but with b+1 multiplying sigma_r**2 instead of just sigma_r**2 
    # because of the form of the intensity rate, b is added to 1 (equation 40 of Ballestra et al.) 
    return q

def E1(params_ce, params_cir, s, q): #q in args to avoid computing it twice, s known, params_cir known
    a, b, c = params_ce # intensity rate parameters, c unused here
    k, mu, r0, sigma_r = params_cir # CIR parameters 

    numerator = 2 * q * np.exp((k - q - (a * sigma_r**2 / (k * mu))) * s / 2)
    denominator = 2 * q + (k - q) * (1 - np.exp(-q * s))
    return (2 * k * mu / sigma_r**2) * np.log(numerator / denominator)

def E2(params_ce, params_cir, s, q):
    a, b, c = params_ce # intensity rate parameters, c unused here
    k, mu, r0, sigma_r = params_cir 
    numerator = -2 * (b + 1) * (1 - np.exp(-q * s))
    denominator = 2 * q + (k - q) * (1 - np.exp(-q * s))
    return numerator / denominator

#%%% CDS pricing CE model
# '''maturities = t_grid'''

def CDS_price_CEa(rec_rate, maturities, params_cir, params_ce, params_gbm):
    ''' This function computes the CDS price using the CE model 
    and gives the term structure of CDS spreads for a specific date.'''
    # CDS pricing using the CEa model

    # PROTECTION LEG
    # find C and D by solving the ODEs
    #s = maturities
    k, mu, r0, sigma_r = params_cir
    a, b, c = params_ce
    x_ratio, alpha, sigma_x = params_gbm #x/xl

    C, D = solve_CD(maturities, params_ce, params_cir, D0=0.0, C0=0.0)
    #print('Estimated C:', C) 
    #print('Estimated D:', D)

    d1 = lambda s : (np.log(x_ratio) + (alpha - 0.5 * sigma_x**2) * s
          ) / (sigma_x * np.sqrt(s))
    d2 = lambda s : (- np.log(x_ratio) + (alpha - 0.5 * sigma_x**2) * s
          ) / (sigma_x * np.sqrt(s))
    protection_leg = ( (1 - 
                        (ndtr( d1(maturities) ) - 
                         np.exp((1-2*alpha/sigma_x**2)*np.log(x_ratio))*ndtr(d2(maturities))
                         ) * np.exp(C + D * r0)
                         ) * (1-rec_rate) 
                         * bond_price_CIR(k,mu,r0,sigma_r,maturities)
    ) 

    #print('Protection Leg:', protection_leg)

    # PREMIUM LEG
    q = find_q(params_ce, params_cir)
    # E1_vals = E1(params_ce, params_cir, s, q) must be specified inside prem fct bc s enters integral
    # E2_vals = E2(params_ce, params_cir, s, q)
    prem_fct = lambda s: (( ndtr(d1(s)) - 
                           np.exp((1-2*alpha/sigma_x**2)*np.log(x_ratio))*ndtr(d2(s))
    ) *np.exp(E1(params_ce, params_cir, s, q) + E2(params_ce, params_cir, s, q)*r0)
    ) 
    premium_leg = [] # works but need faster way 
    for s in maturities:
        prem = integrate.quad(prem_fct,0,s)[0]
        premium_leg.append(prem)

    #print('Premium Leg:', premium_leg)

    price =  protection_leg / premium_leg 
    return protection_leg, premium_leg, price 

#%%% Testing parameters setup
# intensity rate parameters
if __name__ == "__main__": 
    ''' BASE CASE FROM CATHCART, EL-JAHEL 2006 (FIGURE 1) '''

    a = 0.002 #starting value of intensity rate, increases spreads but not massively (a high, spread in 0 higher)
    b = 0.1 #increases spreads massively
    c = 0.0 # set to zero in CEa model

    'Enters CEa model through f(x,t) in survival probability, but GBM (credit quality variable) does NOT affect intensity parameter'
    # GBM parameters 
    x_ratio = 1.5 #x/xl --> under 1 = default , changes nothing?
    alpha = 0.04 # changes nothing? 
    sigma_x = 0.02 # changes from sigma > 0.2

    # CIR parameters
    k_opt = 0.5
    mu_opt = 0.09
    r0_opt = 0.04
    sigma_r_opt = 0.078

    ''' BALLESTRA ET AL. 2020 PARAMS LOW RISK AVRG PARAMS '''
    a = -0.024 #starting value of intensity rate, increases spreads but not massively (a high, spread in 0 higher)
    b = 2.6 #increases spreads massively
    c = -0.032 # set to zero in CEa model

    'Enters CEa model through f(x,t) in survival probability, but GBM (credit quality variable) does NOT affect intensity parameter'
    # GBM parameters 
    x_ratio = 2.85 #x/xl --> under 1 = default , changes nothing?
    alpha = 0.05 # changes nothing? 
    sigma_x = 0.2 # changes from sigma > 0.2

    # CIR parameters
    k_opt = 0.015
    mu_opt = 0.034
    r0_opt = 0.015
    sigma_r_opt = 0.031

    ############ EXAMPLE ############
    params_ce = (a,b,c) #intensity rate parameters
    params_gbm = (x_ratio, alpha, sigma_x) #x_ratio instead of x0 bc it enters as ratio
    params_cir = (k_opt, mu_opt, r0_opt, sigma_r_opt)

    t_grid = np.linspace(1, 10, 5)
    start = time.time()
    print("Time start:", start)
    #CDS_price_CE(0.4, t_grid, params_cir, params_ce, params_gbm)
    protection_leg, premium_leg, spread  = CDS_price_CEa(0.4, t_grid, params_cir, params_ce, params_gbm)
    # E1(params_ce, params_cir, t_grid, find_q(params_ce, params_cir))
    # E2(params_ce, params_cir, t_grid, find_q(params_ce, params_cir))
    
    '''print(f"Protection leg: {protection_leg}")
    print(f"Premium leg: {premium_leg}")
    print(f"CDS price: {price}")     
    '''
    end = time.time()
    print("Time elapsed:", end - start)
    plt.figure(figsize=(7, 4))
    plt.plot(t_grid, spread*10000) #convert in bps
    plt.xlabel("t")
    plt.ylabel("spread")
    plt.title("Term structure")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%

# CEa objective function 
# SETTING VALUES FOR A,B,C ! ESTIMATING ONLY GBM PARAMETERS
def CEa_objective_fct(params_gbm, params_ce, params_cir, rec_rate, market_prices, T):
    x_ratio, alpha, sigma_x = params_gbm
    if x_ratio <= 1 or sigma_x <= 0: #must be positive values
        return np.inf
    else: 
        protection_leg, premium_leg, model_prices = CDS_price_CEa(rec_rate, T, params_cir, params_ce, params_gbm)
        return MAPE(market_prices, model_prices) 
    

def glob_CEa_calibration(params_ce, params_cir, rec_rate, market_prices, T): 
    '''
    NOTICE: 20^3 = 8000 evaluations for each maturity
    '''
    ranges = (
        slice(1.01, 5, complex(20)), #equivalent to np.linspace(a, b, N) 
        # I am placing an upper bound on 5 based off the results in the paper (<=4.5) how to choose otherwise? 
        slice(-0.3, 0.3, complex(20)), # alpha --> should also be possibly NEGATIVE
        slice(0.01, 0.5, complex(20)),) # sigma
    # Global optimization
    res1 = optimize.brute(CEa_objective_fct, 
                          ranges=ranges, 
                          Ns=20, #grid points
                          args=(params_ce,params_cir, rec_rate, market_prices, T),
                          finish=None,
                          workers=-1)
    x_ratio, alpha, sigma_x = res1
    #a_opt_gl, b_opt_gl, r0_opt_gl, sigma_opt_gl = res1

    params = pd.DataFrame(
        {"Global Optimization": [x_ratio, alpha, sigma_x]},
        index=["x_ratio", "alpha", "sigma_x"])
    
    #print("Global optimization results:")
   # display(df_global)

    return params

########## AAAAA
###################### AAAAAAAAAA
######################################## AAAAAAAAAAAAAAA
def loc_CEa_calibration(params_cir, rec_rate, market_prices, T, initial_guess): #market_prices, T, initial_guess): 
    
    #Local optimization  -> CONSTRAINTS ARE IMPORTANT HERE
    res2 = optimize.minimize(CEa_objective_fct, 
                             initial_guess, 
                             method='SLSQP', 
                             bounds=((1.01, None), (None, None), (0.001, None)), 
                             args=(params_ce, params_cir, rec_rate, market_prices, T),
                             #constraints = 
                             )

    
    x_ratio, alpha, sigma_x = res2.x

    params = pd.DataFrame(
        {"Local Optimization": [x_ratio, alpha, sigma_x]},
        index=["x_ratio", "alpha", "sigma_x"])
    # results
    # print("Local optimization results:")
    # display(df_local)

    return params
#%%
if __name__ == "__main__":
    res = CEa_calibration(params_ce, params_cir, 0.4, spread, t_grid)
    loc_res = loc_CEa_calibration(params_cir, 0.4, spread, t_grid, res.values.flatten())
    print(res, loc_res) 
# %%
