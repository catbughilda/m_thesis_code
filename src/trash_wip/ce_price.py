"CE MODEL PRICE"
#%%% CDS pricing under the CE model

import numpy as np
import os
import matplotlib.pyplot as plt

path = '/Users/linaatanasova/Documents/pycourse/qfi/thesis_code'
os.chdir(path)

from src.cir_calibration import bond_price_CIR
from src.num_routines_odes import solve_CD
from trash.laplace import inv_laplace_CE, inv_laplace_CE_termstruct
import scipy.integrate as integrate
import time

# The computations suppose a notional of 1, t_0 = 0, and continuous compounding 

#%%% E1 and E2 functions for CE model
# should i specify the CE_params as params and unpack them inside the function? probably yes

# Auxiliary functions for PREMIUM LEG
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
'''
def find_E1_E2(s, params_ce, params_cir):
    q = find_q(params_ce, params_cir)
    E1_val = E1(params_ce, params_cir, s, q)
    E2_val = E2(params_ce, params_cir, s, q)
    return E1_val, E2_val
'''
# CHECK IF PREMIUM LEG FORMULA IS CORRECT ON PAPER ! ! ! !

#%%% CDS pricing CE model
# '''maturities = t_grid'''

def CDS_price_CE(rec_rate, maturities, params_cir, params_ce, params_gmb): # i'm missing parameters: x_ratio, alpha, sigma_x ...... NOTICE: maturities = s 
    ''' This function computes the CDS price using 
    the CE model and gives the term structure of CDS spreads 
    for a specific date.'''
    # CDS pricing using the CE model

    # PROTECTION LEG
    # find C and D by solving the ODEs
    s = maturities
    k, mu, r0, sigma_r = params_cir
    a,b,c = params_ce
     # solve ODEs to get C(s) and D(s)

    C, D = solve_CD(maturities, params_ce, params_cir, D0=0.0, C0=0.0)
    print('Estimated C:', C) #check 
    print('Estimated D:', D)
    inv_lapl_protection = inv_laplace_CE_termstruct(maturities, params_gmb, params_ce)
    protection_leg = ((1 - inv_lapl_protection * np.exp(C + D * r0)) #r0?
                      * (1 - rec_rate) * bond_price_CIR(k_opt,mu_opt,r0_opt,sigma_r_opt,maturities)
                      ) #time to maturity supposes t=0   # recovery rate delta = 0.4 assumed

    print('Protection Leg:', protection_leg)

    # PREMIUM LEG
    # find E1 and E2
    # E1, E2 = find_E1_E2(maturities, params_ce, params_cir) #E1 and E2 need to be functions that depend on ds
    q = find_q(params_ce, params_cir) # independent of time
    premium_leg = []
    for i, s in enumerate(t_grid):
        #print(np.exp(E1[i] + E2[i] * r0))
        lapl = lambda x: np.exp(E1(params_ce, params_cir, x, q)+ E2(params_ce, params_cir, x, q) * r0) * inv_laplace_CE(x, params_gmb, params_ce)
        prem_leg = integrate.quad(lapl,1e-6,s)[0] #takes only y=  float (itegral from a to b)
             #second argument is abserr = float --> estimate of the absolute error in the result.
        print(prem_leg)
        premium_leg.append(prem_leg) 
    
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
    c = -0.03

    # GBM parameters
    x_ratio = 0.5 #over 1 = default --> CHANGED to x/xl so under 1 = default
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
    print(f"Protection leg: {protection_leg}")
    print(f"Premium leg: {premium_leg}")
    print(f"CDS price: {price}")     
    end = time.time()
    print("Time elapsed:", end - start)

#%% 
plt.figure(figsize=(7, 4))
plt.plot(t_grid, price*100) #convert 
plt.xlabel("t")
plt.ylabel("spread")
plt.title("Term structure")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
