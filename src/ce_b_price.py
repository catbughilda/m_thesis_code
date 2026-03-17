"CEb MODEL:"
"a,b,c = 0"
"Default is structural and depends on a signalling quality variable given by a Geometric Brownian Motion"
#%%% CDS pricing under the CEb model
import os
from dotenv import load_dotenv

load_dotenv()
path = os.getenv('macbook_path')
os.chdir(path)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import ndtr
from scipy import optimize
from src.helper import MAPE
from src.cir_calibration import bond_price_CIR
from src.cir_price_t0_T import bond_price_CIR_t0 #if t0 is different from 0

# The following computations suppose a notional of 1, t_0 = 0, and continuous compounding 

#%%% CDS pricing CEb model
# maturities = t_grid

def CDS_price_CEb(rec_rate, maturities, params_cir, params_gbm): #params_ce,
    ''' 
    Computes the CDS price using the CEb model and gives the term structure of CDS spreads for a specific date.
    The CEb corresponds to a structural model where defaults happens only when X (credit quality)
    goes under a certain value xl.
    '''
    # PROTECTION LEG 
    s = maturities
    k, mu, r0, sigma_r = params_cir
    # a,b,c = params_ce
    x_ratio, alpha, sigma_x = params_gbm #x/xl

    d1 = lambda s : (np.log(x_ratio) + (alpha - 0.5 * sigma_x**2) * s
          ) / (sigma_x * np.sqrt(s))
    d2 = lambda s : (- np.log(x_ratio) + (alpha - 0.5 * sigma_x**2) * s
          ) / (sigma_x * np.sqrt(s))

    protection_leg = ( (1 - 
                        (ndtr(d1(maturities)) - 
                         np.exp((1-2*alpha/sigma_x**2)*np.log(x_ratio))*ndtr(d2(maturities))
                         ) *1) #there is nothing bc lambda is set to 0
                         * (1-rec_rate) 
                         * bond_price_CIR(k,mu,r0,sigma_r,maturities)
    ) 

    #print('Protection Leg:', protection_leg)

    # PREMIUM LEG
    premium_leg = []
    for mat in maturities:
            prem_fct = lambda x: (( ndtr(d1(x)) - 
                           np.exp((1-2*alpha/sigma_x**2)*np.log(x_ratio))*ndtr(d2(x))
                        ) * bond_price_CIR_t0(k,mu,r0,sigma_r,x,mat) # lambda is set to zero!
                        ) 
            prem = integrate.quad(prem_fct,0,mat)[0]
            premium_leg.append(prem)

    #print('Premium Leg:', premium_leg)

    price =  protection_leg / premium_leg 
    #print(price)

    return protection_leg, premium_leg, price 

#%%% Testing parameters setup
# intensity rate parameters
if __name__ == "__main__":
    start = time.time()
    print("Time start:", start)
    a = 0
    b = 0
    c = 0 # set to zero in ce_b model

    # GBM parameters
    x_ratio = 1.5 # x/xl => under 1 = default 
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

    t_grid = np.linspace(1, 10, 10)
    #CDS_price_CE(0.4, t_grid, params_cir, params_ce, params_gbm)
    protection_leg, premium_leg, price  = CDS_price_CEb(0.4, t_grid, params_cir, params_gbm)
    '''print(f"Protection leg: {protection_leg}")
    print(f"Premium leg: {premium_leg}")
    print(f"CDS price: {price}")     
    '''
    end = time.time()
    print("Time elapsed:", end - start)
    plt.figure(figsize=(7, 4))
    plt.plot(t_grid, price*10000) #convert to bps
    plt.xlabel("t")
    plt.ylabel("spread")
    plt.title("Term structure")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%

# CEb objective function 

def CEb_objective_fct(params_gbm, params_cir, rec_rate, market_prices, T):
    x_ratio, alpha, sigma_x = params_gbm

    if x_ratio <= 0 or sigma_x <= 0: #must be positive values
        return np.inf
    else: 
        protection_leg, premium_leg, model_prices = CDS_price_CEb(rec_rate, T, params_cir, params_gbm)
        return MAPE(market_prices, model_prices) 
    

# Global optimization
# very sensitive to bounds specification 
def glob_CEb_calibration(params_cir, rec_rate, market_prices, T): 
    '''
    NOTICE: 20^3 = 8000 evaluations for each maturity
    '''
    ranges = (
        slice(1.01, 5, complex(20)), #equivalent to np.linspace(a, b, N) 
        # I am placing an upper bound on 5 based off the results in the paper (<=4.5) how to choose otherwise? 
        slice(-0.3, 0.3, complex(20)), # alpha --> should also be possibly NEGATIVE
        slice(0.01, 0.5, complex(20)),) # sigma
    # Global optimization
    res1 = optimize.brute(CEb_objective_fct, 
                          ranges=ranges, 
                          Ns=20, #grid points
                          args=(params_cir, rec_rate, market_prices, T),
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

def loc_CEb_calibration(params_cir, rec_rate, market_prices, T, initial_guess): #market_prices, T, initial_guess): 
    
    #Local optimization for refinement
    res2 = optimize.minimize(CEb_objective_fct, 
                             initial_guess, 
                             method='Nelder-Mead', 
                             bounds=((1.01, 5), (-0.3, 0.3), (0.01, 0.5)), #x_ratio, alpha, sigma_x must be positive
                             args=(params_cir, rec_rate, market_prices, T))
    
    x_ratio, alpha, sigma_x = res2.x

    params = pd.DataFrame(
        {"Local Optimization": [x_ratio, alpha, sigma_x]},
        index=["x_ratio", "alpha", "sigma_x"])
    # results
    # print("Local optimization results:")
    # display(df_local)

    return params

def CEb_calibration(params_cir, rec_rate, market_prices, T):
    glob_res = glob_CEb_calibration(params_cir, rec_rate, market_prices, T)
    loc_res = loc_CEb_calibration(params_cir, rec_rate, market_prices, T, glob_res.values.flatten())
    return loc_res, glob_res


# %%
if __name__ == "__main__":
    x_ratio = 1.5 # x/xl => under 1 = default 
    alpha = 0.05
    sigma_x = 0.2
    initial_guess = (x_ratio, alpha, sigma_x)
    #res = loc_CEb_calibration(params_cir, 0.4, price, t_grid, initial_guess)
    #res
    res = CEb_calibration(params_cir, 0.4, price, t_grid)
    print(res) 
# %%
