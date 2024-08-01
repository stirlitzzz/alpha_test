from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame
import FMNM

def test_func(b):
    result=b*2
    return result
def d1(S,K,T,r,sigma):
    return (np.log(S/K) + (r + sigma**2/2.) * T) / (sigma * np.sqrt(T))

def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma) - sigma * np.sqrt(T)

def bs_call(S,K,T,r,sigma):
    return S*norm.cdf(d1(S,K,T,r,sigma))-K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))

from FMNM.BS_pricer import BS_pricer
from FMNM.Parameters import Option_param
from FMNM.Processes import Diffusion_process
from FMNM.cython.solvers import PSOR

import numpy as np
import scipy.stats as ss

import matplotlib.pyplot as plt

%matplotlib inline
from IPython.display import display
import sympy

sympy.init_printing()


S=np.array([100,100])
K=np.array([100,120])
T=np.array([1,1])
r=np.array([0.0,0.])
sigma=np.array([0.2,0.2])
print(bs_call(S,K,T,r,sigma))
