#%% CIR Calibration
import pandas as pd
import numpy as np
import math
from scipy import optimize
from matplotlib import pyplot as plt
from src.helper import MAPE

"""CIR Calibration"""

#%%% Functions

# Notice: in Ballestra et al. (2007) the parameters are a, b, sigma are called kappa, mu, sigma_r

# Bond price CIR model
def A_CIR(a,b,sigma,T):
        h = math.sqrt(a**2 + 2*sigma**2)
        numerator_A = 2*h*np.exp((a+h)*T/2)
        denominator_A = 2*h + (a+h)*(np.exp(h*T)-1)
        A = (numerator_A/denominator_A)**(2*a*b/sigma**2)
        return A

def B_CIR(a,sigma,T): # needed also in CDS price fct 
        h = math.sqrt(a**2 + 2*sigma**2)
        B = 2*(np.exp(h*T)-1)/(2*h + (a+h)*(np.exp(h*T)-1))
        return B

def bond_price_CIR(a,b,r0,sigma,T): # supposing t=0, we only need maturity T
        A = A_CIR(a,b,sigma,T)
        B = B_CIR(a,sigma,T)
            
        return A * np.exp(-B*r0) # should be an np.array 

# CIR bjective function
def CIR_objective_fct(params, market_prices, T):
    a, b, r0, sigma = params
    if a <= 0 or r0 <= 0 or sigma <= 0: #must be positive values
        return np.inf
    if 2*a*b < sigma**2: #Feller condition for non-zero interest rates
        return np.inf
    else:
        model_prices = bond_price_CIR(a,b,r0,sigma,T)
        return MAPE(model_prices, market_prices) 

# Global optimization 
def glob_CIR_calibration(market_prices, T): 
    '''
    NOTICE: int_rate is a dataframe with columns 'y_frac' and 'market_price'
    '''
    bounds = [(0.00001, 1), (0.00001, 1), (0.00001, 0.1), (0.00001, 1)]
    # Global optimization
    res1 = optimize.brute(CIR_objective_fct, 
                          ranges=bounds, 
                          Ns=20, #grid points
                          args=(market_prices, T),)
    a_opt_gl, b_opt_gl, r0_opt_gl, sigma_opt_gl = res1

    params = pd.DataFrame(
        {"Global Optimization": [a_opt_gl, b_opt_gl, r0_opt_gl, sigma_opt_gl]},
        index=["a", "b", "r0", "sigma"])
    
    #print("Global optimization results:")
   # display(df_global)

    return params

# Local optimization
def loc_CIR_calibration(market_prices, T, initial_guess): 
    '''
    NOTICE: int_rate is a dataframe with columns 'y_frac' and 'market_price'
    initial_guess is a list or np.array with initial parameters [a, b, r0, sigma]
    '''
    
    #Local optimization for refinement
    res2 = optimize.minimize(CIR_objective_fct, 
                             initial_guess, 
                             method='Nelder-Mead',
                             bounds = [(1e-5, 1), (1e-5, 1), (1e-8, 0.1), (1e-5, 1)], 
                             args=(market_prices, T) )
    a_opt, b_opt, r0_opt, sigma_opt = res2.x

    params = pd.DataFrame(
        {"Local optimization": [a_opt, b_opt, r0_opt, sigma_opt]},
        index = ["a", "b", "r0", "sigma"] # k, mu, r0, sigma_r
    )
    # results
    # print("Local optimization results:")
    # display(df_local)

    return params

#%%% Calibration

def CIR_calibration(market_prices, T):
    global_res = glob_CIR_calibration(market_prices, T)
    local_res = loc_CIR_calibration(market_prices, T, global_res.values.flatten())
    return global_res, local_res


#%%% Calibration plot

def CIR_plot(int_rate, params_gl, params_loc):
    a_gl, b_gl, r0_gl, sigma_gl = params_gl
    a_lc, b_lc, r0_lc, sigma_lc = params_loc

    T = int_rate['y_frac']
    market_prices = int_rate['market_price']

    model_prices_gl = bond_price_CIR(a_gl, b_gl, r0_gl, sigma_gl, T)
    model_prices_lc = bond_price_CIR(a_lc, b_lc, r0_lc, sigma_lc, T)

    plt.figure(figsize=(10, 6))
    plt.plot(T, model_prices_gl, label="Global CIR", linewidth=4)
    plt.plot(T, model_prices_lc, label="Local CIR (adj.)", linewidth=2)
    plt.scatter(T, market_prices, label="Market Prices", color="grey", linewidth=1)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Bond Price")
    plt.title("CIR Calibration")
    plt.legend()
    plt.grid()
    plt.show()

#%%% Example from Damiano, Brigo 2006

if __name__ == "__main__":
    # maturities in years
    T = np.array([0.25, 0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    # market zero-coupon yields (continuously compounded)
    y = np.array([0.0200, 0.0220, 0.0250, 0.0300, 0.0340, 0.0400, 0.0430, 0.0450])
    int_rate = pd.DataFrame({

        "int_rate": y,
        "market_price": np.exp(-y*T),
        "y_frac": T,
    })
    print(int_rate)
    # Calibration
    global_params = glob_CIR_calibration(int_rate['market_price'].values, int_rate['y_frac'].values)
    local_params = loc_CIR_calibration(int_rate['market_price'].values, int_rate['y_frac'].values, global_params.values.flatten())  
    print("Global Optimization Parameters:")
    print(global_params)
    print("Local Optimization Parameters:")
    print(local_params)             
    CIR_plot(int_rate, global_params.values.flatten(), local_params.values.flatten())
    '''
    # A, B cir
    a, b, r0, sigma = local_params.values.flatten()
    s_grid = np.linspace(0, 10, 100)
    A_values = A_CIR(a, b, sigma, s_grid)
    B_values = B_CIR(a, sigma, s_grid)
    plt.subplot(2, 1, 1)
    plt.plot(s_grid, A_values, label="A(s)", linewidth=2)   
    plt.subplot(2, 1, 2)
    plt.plot(s_grid, B_values, label="B(s)", linewidth=2)
    plt.xlabel("Maturity (s)")        
    '''
# %%
