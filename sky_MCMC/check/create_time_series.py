import numpy as np
import pandas as pd
import scipy.special as special
from scipy.special import jv

au = 1.496e+13
solar_mass = 1.989e+33
G = 6.67e-8
C_CGS = 3e+10


def kepler_series_solution(M, e, terms=20):

    """Series solution of the kepler equation to reduce time
    """
    E = M
    for n in range(1, terms + 1):
        E += (2 / n) * jv(n, n * e) * np.sin(n * M)
    return E

def series(constants,n):  
    """The cost function which should be minimized for
       the correct inference of the orbital parameters
    """

    t_start=constants[0]  # in term of 1e6
    a_out=constants[1]     #in terms of 1000
    w_out=constants[2]
    e=constants[3]
    time_in=constants[4]   #in 1e6
    M_out= 0.4
    i_in=constants[5]
    theta_s=constants[6]
    phi_s=constants[7]

    T_lisa=31556952  # in seconds 
    
    T_cal = np.zeros(n)
    a = a_out*au

    # Normalizing the values of the constant
    t_start*=1e6
    a*=1000
    time_in*=1e6
    M_out*=1e7

    for i in range(n):
# Roemer Delay due to revolution of Binary around the SBH

        t0=t_start+i*(time_in)
        M = t0 * np.sqrt((G * M_out * solar_mass / (np.power(a,3)))) 
        M_range=(M + np.pi) % (2 * np.pi) - np.pi
        E=kepler_series_solution(M_range,e)  
        beta = e / (1 + np.sqrt(1 - e ** 2))
        F=E + 2 * np.arctan(beta * np.sin(E) / (1 - beta * np.cos(E)))
        R=(a * (1 - e** 2) / (1 + e * np.cos(F))) / C_CGS
        Doppler = 2 * np.pi * R * np.sin(F + w_out) * np.sin(i_in)
        t_calculated = Doppler / (2 * np.pi)
        t_calculated = (time_in*i) - t_calculated
  
# Roemer Delay due to the LISA's orbit

        RD_LISA = 499.00478 * np.sin(theta_s) * np.cos((2 * np.pi * (i*time_in/T_lisa))-phi_s)
        T_cal[i] = t_calculated - RD_LISA
    T_cal-=T_cal[0]
    return T_cal
