"GOODNESS OF FIT"

import pandas as pd
import numpy as np
import math
from scipy import optimize
from matplotlib import pyplot as plt

# MAPE (Mean Absolute Percentage Error)
def MAPE(market_prices, model_prices):
    return 1/len(market_prices) * np.sum(np.abs((market_prices - model_prices)/market_prices)) 

# why do we choose MAPE? because it gives a relative error, so it is not biased towards high or low prices, 
# so it avoids overfitting the long term CDS prices (usually the highest)
# we are testing whether the model replicates the prices closely, 
# so using Root Mean Squared Percentage Error (RMSPE) would not be appropriate,
# because large errors will be squared and thus given more weight in the final error metric
# these erros come from a single maturity, so the model is overfitting to one specific maturity
# instead of replicating the whole term structure of CDS spreads and giving the same importance even to small errors if present across different maturities
# It could be interesting to see how the calibrations would behave using RMSPE instead of MAPE, also suggested in the paper.
