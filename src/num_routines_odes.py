#%%
import numpy as np
import os
import matplotlib.pyplot as plt
path = '/Users/linaatanasova/Documents/pycourse/qfi/thesis_code'
os.chdir(path)
from scipy.integrate import solve_ivp
from src.cir_calibration import B_CIR

# We need to solve the ODE system for D(s) and C(s):

 #   D'(s) = 0.5 * σ_r^2 * D(s)^2 + (σ_r^2  * B(s) - k) * D(s) - b
 #   C'(s) = k * μ * D(s) - a
    
 #   with initial conditions D(0) = 0, C(0) = 0, B(s) given from the CIR model.

#%%% Defining the system of ODEs --> y = [D(s), C(s)]

def ode_system(s, y, params_ce, params_cir): # s = maturities, y = [D(0), C(0)]
    D, C = y # D at specific maturity
    a, b, c = params_ce # intensity rate parameters, c unused here
    k, mu, r0, sigma_r = params_cir # CIR parameters 

    # diff equation for D(s)
    D_s = ( #equation A14 in Cathcart El-Jahel 2006
        0.5 * sigma_r**2 * D**2
        + (sigma_r**2 * B_CIR(k,sigma_r,s) - k) * D
        - b # b is unknown 
    )

    # diff equation for C(s)
    C_s = k * mu * D - a # a is unknown constant from the intesnsity rate

    return [D_s, C_s] #derivatives

#%%% Solver function
def solve_CD(s_grid, params_ce, params_cir, D0=0.0, C0=0.0): #initial conditions are given
    """
    Solve for D(s) and C(s) on the supplied grid s_grid.

    Parameters 
    s_grid : increasing np.array of maturities
    D0 : initial FLOAT value of D at s = s_grid[0]
    C0 : initial FLOAT value of C at s = s_grid[0]

    Output
    D_vals : np.array --> D(s) values
    C_vals : np.array --> C(s) values
    """
    y0 = [D0, C0]

    sol = solve_ivp(
        ode_system, # the system of ODEs in one single fct
        t_span=(0, s_grid[-1]), #first and last value of the grid
        y0=y0, #initial conditions
        t_eval=s_grid, #points where we want the solution --> maturities grid
        args=(params_ce, params_cir), #parameters for the ODE system
        method="RK45", #default method --> Runge-Kutta 5(4)
    )
    #### As for the method: ‘RK45’ (default): Explicit Runge-Kutta method of order 5(4)
    # The error is controlled assuming accuracy of the fourth-order method, 
    # but steps are taken using the fifth-order accurate formula (local extrapolation is done). 
    # A quartic interpolation polynomial is used for the dense output.
    # Can be applied in the complex domain. (source: docs.scipy.org)

    #FROM CATHCART EL-JAHEL 2006: 
    # Equation (A14) is of the Ricatti type, we solve it using the Runge–Kutta method with initial conditions D(0) = 0 and D'(0) = b.
    # WHY b and not -b ?

    if not sol.success:
        raise RuntimeError("ODE solver failed: " + sol.message)

    D_vals = sol.y[0]
    C_vals = sol.y[1]

    return D_vals, C_vals #np.arrays with solutions for the different maturities

#%%% Example 
if __name__ == "__main__":
    sigma_r = 0.02       # volatility of r
    k = 0.30             # speed of mean reversion
    mu = 0.05            # long-term mean
    a = 0.01             # constant term in intensity rate
    b = 0.02            # coefficient of r in intensity rate
    r0 = 0.03            # short rate

    params_ce = (a, b, 0.0)  # c is unused here and rest of params are unused
    params_cir = (k, mu, r0, sigma_r)
    # Example maturity grid
    s = np.linspace(0, 10, 100)

    # Solve system
    D_solution, C_solution = solve_CD(s, params_ce=params_ce, params_cir=params_cir)

    # Print or save results
    print("D(s) =")
    print(D_solution)

    print("\nC(s) =")
    print(C_solution)

    plt.subplot(2, 1, 1)
    plt.plot(s, D_solution, label="D(s)")
    plt.title("Solution of D(s)")
    plt.xlabel("Maturity s")
    plt.ylabel("D(s)")
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(s, C_solution, label="C(s)", color="orange")
    plt.title("Solution of C(s)")
    plt.xlabel("Maturity s")
    plt.ylabel("C(s)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
