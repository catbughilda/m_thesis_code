import math
import numpy as np

def A_CIR(a,b,sigma,t0,T):
        h = math.sqrt(a**2 + 2*sigma**2)
        numerator_A = 2*h*np.exp((a+h)*(T-t0)/2)
        denominator_A = 2*h + (a+h)*(np.exp(h*(T-t0))-1)
        A = (numerator_A/denominator_A)**(2*a*b/sigma**2)
        return A

def B_CIR(a,sigma,t0,T): # needed also in CDS price fct 
        h = math.sqrt(a**2 + 2*sigma**2)
        B = 2*(np.exp(h*(T-t0))-1)/(2*h + (a+h)*(np.exp(h*(T-t0))-1))
        return B

def bond_price_CIR_t0(a,b,r0,sigma,t0,T): # supposing t=0, we only need maturity T
        A = A_CIR(a,b,sigma,t0,T)
        B = B_CIR(a,sigma,t0,T)
            
        return A * np.exp(-B*r0)