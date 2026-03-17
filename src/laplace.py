""" LAPLACE """
# %%%%
''' f(x,tau) defined for tau >= 0, 
    with generic Laplace transform defined as:
        F (x,q) = L{f(x,q)} = int_0^infty [ e^{-q*s} * f(x,s) ] ds
        (notice q = Laplace parameter, we go from t to q space)
    and in particular we have explicit form:
       F (x,q) = expression (13) in Ballestra et al. (2020)'''

''' Other: 
    functions need to take array/list t_grid as input ! 
    Do I invert for each point on t-grid? Depends on method I choose.s'''

#import numpy as np
#from scipy.special import iv, gamma, hyp1f1
import numpy as np
import matplotlib.pyplot as plt
from mpmath import hyp1f2, besseli, gamma, invertlaplace, mp,  sin, pi, sqrt, hyper
#from math import sin, pi, sqrt
### decided to use mpmath because all functions are available
# creating mpmath numbers to use in mpmath functions, but they are not compatible with numpy functions
# to solve numpy/mpmath compatibility i convert final results in floats !!!!!

# Why Stehfest? bc it works well with smooth monotone functions
mp.dps = 50
#%%% Laplace for CE model

def laplace_CE(q, params_gmb, params_ce,): # params_cir):
                #____lambda0____, a, b, c, x_ratio, alpha, sigma_x,): #
    ''' Laplace transform for the CE model as in Ballestra et al. (2020), 
        x_ratio = x_l / x, term seen also in intensity process
        q = Laplace parameter'''
    #k, mu, r0, sigma_r  = params_cir
    a, b, c = params_ce
    x_ratio, alpha, sigma_x = params_gmb #x_ratio not x_0 --> x_l/x

    ''' Laplace transform for the CE model as in Ballestra et al. (2020)'''
    eta = sqrt(8) / sigma_x
    zeta = alpha / sigma_x**2 - 0.5
    gam = 0.5
    n = 2*sqrt(zeta**2 + 2*q / sigma_x**2) #depends on q
    #n = min(n, 50)
    xi = - (zeta + gam) / gam

    # HELPERS
    #modified Bessel functions of the first kind
    iv_arg_cx = eta * (c * x_ratio)**gam
    iv_arg_c = eta * c**gam

    bessel_n_cx = besseli(n, iv_arg_cx) # i use n in the name but refers to eta
    bessel_mn_cx = besseli(-n, iv_arg_cx)
    bessel_mn_c = besseli(-n, iv_arg_c)
    bessel_n_c = besseli(n, iv_arg_c)

    nxi = 1 + xi + n
    mnxi = 1 + xi - n
    nxi3 = 3 + xi + n
    mnxi3 = 3 + xi - n
    n1 = 1 + n
    mn1 = 1 - n

    #gamma functions
    gamma_nxi = gamma(nxi/2)
    gamma_mnxi = gamma(mnxi/2)
    gamma_n1 = gamma(n1)
    gamma_mn1 = gamma(mn1)
    gamma_nxi3 = gamma(nxi3/2)
    gamma_mnxi3 = gamma(mnxi3/2)

    # hypergeometric functions
    # mpmath hyp1f2 better convergence on large arguments
    # hyp1f2(a, b1, b2, z)
    def H(a,b1,b2,z):
        return mp.hyper([a], [b1, b2], z)
    
    hyp1f2_pos_c = H(nxi/2, n1, nxi3/2, iv_arg_c**2 / 4 )#hyp1f2(nxi/2, n1, nxi3/2, iv_arg_c**2 / 4 )
    hyp1f2_neg_c = H(mnxi/2, mn1, mnxi3/2, iv_arg_c**2 / 4 )
    hyp1f2_pos_cx = H(nxi/2, n1, nxi3/2, iv_arg_cx**2 / 4 )
    hyp1f2_neg_cx = H(mnxi/2, mn1, mnxi3/2, iv_arg_cx**2 / 4 )

    # PREFAC and TERMS

    prefactor = ( 
        (2**(-1-n) * pi * eta**(zeta/gam) * (c * x_ratio)**zeta) / 
        (sigma_x**2 * gam**2 * sin(pi * n)) 
                 )
    
    term_1 = ( ( bessel_n_cx 
                  * bessel_mn_c 
                  * iv_arg_c**nxi
                  * gamma_nxi 
                  * hyp1f2_pos_c 
                  )
                / 
                ( bessel_n_c
                * gamma_n1 
                * gamma_nxi3
                )
    )

    term_2 = (
        ( 2**(2*n) 
          * bessel_n_cx 
          * gamma_mnxi 
          * iv_arg_cx**mnxi 
          * hyp1f2_neg_cx 
        )
        /
        ( gamma_mn1 
          * gamma_mnxi3 
        )
    )

    term_3 = (
        ( 2**(2*n) 
          * bessel_n_cx 
          * gamma_mnxi 
          * iv_arg_c**mnxi 
          * hyp1f2_neg_c 
        )
        /
        ( gamma_mn1 
          * gamma_mnxi3 
        )
    )

    term_4 = (
        ( bessel_mn_cx 
          * iv_arg_cx**nxi 
          * gamma_nxi 
          * hyp1f2_pos_cx 
        )
        /
        ( gamma_n1 
          * gamma_nxi3 
        )
    )

    laplace_CE = prefactor * ( - term_1 - term_2 + term_3 + term_4 )
    return laplace_CE


#################### INVERSE LAPLACE TRANSFORM ##############################
#%%% Inverse Laplace transform for CE model
mp.dps = 80  # set decimal places for higher precision 

def inv_laplace_CE(t, params_gmb, params_ce):
    ''' Inverse Laplace transform for the CE model as in Ballestra et al. (2020) using Gaver-Stehfest algorithm
    t: time point where to evaluate f(t)
    F: Laplace transform function F(q) that needs to be inverted
    
    returns f(t) evaluated at time t
    '''
    F = lambda q: laplace_CE(q, params_gmb, params_ce)
    return invertlaplace(F, t, method='stehfest') # 'stehfest' 'talbot' 'dehoog' methods available


def inv_laplace_CE_termstruct(t_grid, params_gmb, params_ce):
    ''' Stehfest value for q gets recomputed each time based on t! 
    t_grid : list/array of maturities --> MUST BE ITERABLE !
    other parameters : needed in inv_laplace
    '''
    ##### CAN THIS BE DONE FASTER????? XXXXXXXXXXXXXXXXXX
    f_vals = []
    for t in t_grid:
        try:
            val = inv_laplace_CE(t, params_gmb, params_ce)
            f_vals.append(float(val))  # need floats for plotting and coherence with rest of the functions
        except Exception as e:
            print(f"Failed at t={t}: {e}")
            f_vals.append(np.nan)
    return np.array(f_vals)
    

#%%% HELPER GRAPH FOR INVERSE LAPLACE

import numpy as np
import matplotlib.pyplot as plt

def plot_inv_laplace_CE(t_grid, f_vals):
    """
    Helper plot for inverse Laplace transform of the CE model.
    Evaluates f(t) on a grid and plots it.
    """

    plt.figure(figsize=(7, 4))
    plt.plot(t_grid, f_vals)
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title("Inverse Laplace transform – CE model (Stehfest)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%% TESTING
if __name__ == "__main__":
    params_gmb = (0.4, -0.05, 0.18)  # x_ratio, alpha, sigma_x
    params_ce = (0.002, 2.6, -0.03)   # a, b, c

    #one value
    print(
        f"Inverse Laplace transform for CE model at t=1.0: "
        f"{inv_laplace_CE(1.0, params_gmb, params_ce)}"
    )

    # helper graph
    t_grid = np.linspace(1, 20, 100)
    f_vals = inv_laplace_CE_termstruct(t_grid,params_gmb,params_ce)
    plot_inv_laplace_CE(t_grid, f_vals)
    
    ### graph is a bit weird, highly sensitive on parameters and it doesn't seem monotonic


# %%
## EXAMPLE with known f(t)
run = False 

if run: 
    # Set precision
    mp.dps = 50

    # Define Laplace function F(s)
    def F(s, x):
        return x/(s + 2)

    # Define time points to evaluate inverse Laplace
    t_vals = np.linspace(0.001, 5, 20)

    # Compute numerical inverse Laplace using Stehfest
    def invert_laplace_F(s, x):
        F_help = lambda q: F(q, x)
        return invertlaplace(F_help, s, method='stehfest')

    f_vals = [invert_laplace_F(t, 1.0) for t in t_vals]

    # Convert to floats for plotting
    f_vals = np.array([float(f) for f in f_vals])

    # Exact function for comparison
    f_exact = np.exp(-2 * t_vals)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(t_vals, f_vals, label='Stehfest numerical inversion')
    plt.plot(t_vals, f_exact, '--', label='Exact f(t) = exp(-2t)')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title('Inverse Laplace Transform using Stehfest Method')
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
