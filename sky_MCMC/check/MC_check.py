#!/usr/bin/en
# coding: utf-8


import os
import emcee
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy.special import jv
from scipy import optimize
from scipy.special import jn
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import h5py
import corner
import multiprocessing
import sys
import create_time_series

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


#T_burst=np.loadtxt("time_data/alan_new_data.dat")
cons=[1,0.1,0,0.2,0.2592,0.8,0.1,0.6]
T_burst=create_time_series.series(cons,200)
print(T_burst)
n=len(T_burst)


def to_minimize(constants):  
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
    n=len(T_burst)
    
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

    sigma=60
    T_cal = T_cal - T_cal[0]
    cost = np.sum((T_cal - T_burst) ** 2)/(2*(sigma**2))
    return cost     

#T_burst=create_time_burst(t_start=2e5,n=300,a_out=200,w_out=np.pi/4,time_in=259200,e=0.4,M_out=4.5e+6,i_=np.pi/6)

bounds = [(-30, 30),  
    (0.05, 0.4),
    (-np.pi,np.pi),     
    (0.0,0.3),    
    (0.0, 0.5),
    (-np.pi/2,np.pi/2),
    (-np.pi,np.pi),
    (0,np.pi/2)]

def log_prob(params):
    """ The likelihood function which should be maximized to get
        the correct posterior
    """
    for i, (lower, upper) in enumerate(bounds):
        if not (lower <= params[i] <= upper):
            return -np.inf
   
    cost = to_minimize(params)
    return -cost


ndim = 8
nwalkers = 128
initial_guess = np.random.rand(ndim)

"""First using differential evolution to get an estimate. The goal here is
to provide the MCMC with a small hypersphere as initial parameters which will
explore the whole parameter space"""

result = differential_evolution(to_minimize, bounds,strategy="best1bin",maxiter=15)

#creating that hyper-sphere

pos = result.x + 1e-4* np.random.randn(nwalkers, ndim)

#Using pool for using of multiple CPUs 

pool = multiprocessing.Pool()

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)

#Defing number of steps
nsteps = 50000

sampler.run_mcmc(pos, nsteps, progress=True,store=True)

#Acceptance ratio gives us a good estimate about the health of the MCMC we just ran
print("Acceptance ratio\n",sampler.acceptance_fraction)

#Exception handling for the case when model cannot give the estimate of autocorrelation time
#if such situation arises use broad or different prior and increase the number of steps
try:
   tau = sampler.get_autocorr_time()
   burnin = int(2 * np.max(tau))
   thin_ = int(0.5 * np.min(tau))

except Exception as e:
    print(f"Error calculating autocorrelation time: {e}")
    print(f"Error calculating autocorrelation time: {e}")
    tau=2000
    burnin=4000
    thin_=200


samples = sampler.get_chain(discard=burnin, flat=True, thin=thin_)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin_)
log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin_)
print(sampler.acceptance_fraction)


print("burn-in: {0}\n".format(burnin))
print("thin: {0}\n".format(thin_))
print("flat chain shape: {0}\n".format(samples.shape))
print("flat log prob shape: {0}\n".format(log_prob_samples.shape))

labels = ["time_{start}","a_{out}","w_{out}","e_{out}","time_{in}","i","theta_source","phi_source"]

samples_array = np.array(samples)
file_path="check"
if not os.path.exists(file_path):
    os.makedirs(file_path)

# Save the NumPy array to a text file in the specified directory
rname="Result_1"
file_path = os.path.join(file_path, rname+".txt")
np.savetxt(file_path, samples_array)

print(f'Samples saved to {file_path}')

#Saving the corner plot for this run
corner.corner(samples, labels=labels)
plt.savefig("Result with source localization/"+rname+".png")
