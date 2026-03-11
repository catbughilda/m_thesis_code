"CE MODEL PRICE"
#%%% CDS pricing under the CE model

import numpy as np
import os
import matplotlib.pyplot as plt

path = '/Users/linaatanasova/Documents/pycourse/qfi/thesis_code'
os.chdir(path)

from src.cir_calibration import bond_price_CIR
from src.num_routines_odes import solve_CD
#from src.laplace import inv_laplace_CE, inv_laplace_CE_termstruct
import scipy.integrate as integrate
from scipy.special import ndtr
import time

# brute used with up to 3 parameters, need better glob solver
# scipy.optimize -> DIFFERENTIAL EVOLUTION ?


# The computations suppose a notional of 1, t_0 = 0, and continuous compounding 
#%% 
def find_q(params_ce, params_cir): #params_ce are the parameters of the intensity rate
    a, b, c = params_ce # intensity rate parameters, c unused here
    k, mu, r0, sigma_r = params_cir # CIR parameters from previous optimization

    q = np.sqrt(k**2 + 2*(b+1)*sigma_r**2) # b is unknown, k and sigma_r from CIR calibration
    return q

def E1(params_ce, params_cir, s, q): #q in args to avoid computing it twice, s known, params_cir known
    a, b, c = params_ce # intensity rate parameters, c unused here
    k, mu, r0, sigma_r = params_cir # CIR parameters 

    numerator = 2 * q * np.exp((k - q - a * sigma_r**2 / (k * mu)) * s / 2)
    denominator = 2 * q + (k - q) * (1 - np.exp(-q * s))
    return (2 * k * mu / sigma_r**2) * np.log(numerator / denominator)

def E2(params_ce, params_cir, s, q):
    a, b, c = params_ce # intensity rate parameters, c unused here
    k, mu, r0, sigma_r = params_cir # CIR parameters --> CHECK ORDER ! ! ! ! !
    numerator = -2 * (b + 1) * (1 - np.exp(-q * s))
    denominator = 2 * q + (k - q) * (1 - np.exp(-q * s))
    return numerator / denominator

#%%% CDS pricing CE model
# '''maturities = t_grid'''

def CDS_price_CE(rec_rate, maturities, params_cir, params_ce, params_gmb):
    ''' This function computes the CDS price using the CE model and gives the term structure of CDS spreads for a specific date.'''
    # CDS pricing using the CE model

    # PROTECTION LEG
    # find C and D by solving the ODEs
    s = maturities
    k, mu, r0, sigma_r = params_cir
    a,b,c = params_ce
    x_ratio, alpha, sigma_x = params_gmb #xl/x

    C, D = solve_CD(maturities, params_ce, params_cir, D0=0.0, C0=0.0)
    print('Estimated C:', C) #check 
    print('Estimated D:', D)

    d1 = lambda s : (np.log(x_ratio**(-1)) + (alpha - 0.5 * sigma_x) * s
          ) / sigma_x * np.sqrt(s)
    d2 = lambda s : (- np.log(x_ratio**(-1)) + (alpha - 0.5 * sigma_x) * s
          ) / sigma_x * np.sqrt(s)

    protection_leg = ( (1 - 
                        (ndtr(d1(maturities)) - 
                         np.exp((1-2*alpha/sigma_x**2)*np.log(x_ratio**(-1)))*ndtr(d2(maturities))
                         ) * np.exp(C + D * r0)) 
                         * (1-rec_rate) 
                         * bond_price_CIR(k_opt,mu_opt,r0_opt,sigma_r_opt,maturities)
    ) 

    print('Protection Leg:', protection_leg)

    # PREMIUM LEG
    q = find_q(params_ce, params_cir)

    prem_fct = lambda s: (( ndtr(d1(s)) - 
                           np.exp((1-2*alpha/sigma_x**2)*np.log(x_ratio**(-1)))*ndtr(d2(s))
    ) *np.exp(E1(params_ce, params_cir, s, q) + E2(params_ce, params_cir, s, q)*r0)
    ) 
    premium_leg = [] # works but need faster way 
    for s in maturities:
        prem = integrate.quad(prem_fct,0,s)[0]
        premium_leg.append(prem)

    print('Premium Leg:', premium_leg)

    price =  protection_leg / premium_leg 
    return protection_leg, premium_leg, price 

#%%% Testing parameters setup
# intensity rate parameters
if __name__ == "__main__":
    start = time.time()
    print("Time start:", start)
    a = 0.02
    b = 2.6
    c = 0 # set to zero in ce_b model

    # GBM parameters
    x_ratio = 0.4 #over 1 = default 
    alpha = 0.05
    sigma_x = 0.2

    # 
    k_opt = 0.03
    mu_opt = 0.02
    r0_opt = 0.01
    sigma_r_opt = 0.015

    params_ce = (a,b,c) #intensity rate parameters
    params_gbm = (x_ratio, alpha, sigma_x) #x_ratio instead of x0 bc it enters as ratio
    params_cir = (k_opt, mu_opt, r0_opt, sigma_r_opt)

    t_grid = np.linspace(1, 10, 5)
    #CDS_price_CE(0.4, t_grid, params_cir, params_ce, params_gbm)
    protection_leg, premium_leg, price  = CDS_price_CE(0.4, t_grid, params_cir, params_ce, params_gbm)
    '''print(f"Protection leg: {protection_leg}")
    print(f"Premium leg: {premium_leg}")
    print(f"CDS price: {price}")     
    '''
    end = time.time()
    print("Time elapsed:", end - start)
    plt.figure(figsize=(7, 4))
    plt.plot(t_grid, price*100) #convert 
    plt.xlabel("t")
    plt.ylabel("spread")
    plt.title("Term structure")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
