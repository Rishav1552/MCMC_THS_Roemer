#!/usr/bin/en

from scipy import optimize
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

au = 1.496e+13
solar_mass = 1.989e+33
G = 6.67e-8
C_CGS = 3e+10
T_lisa=31556952  # in seconds 

def func_to_sol(E, M,e):
        return E - e * np.sin(E) - M

def der_to_func(E, M,e):
        return 1 - e  * np.cos(E)

def double_der_to_func(E, M,e):
        return e * np.sin(E)

def solve_kepler(M,e):

        E = np.pi  
        E = optimize.newton(func_to_sol, E, args=(M,e), fprime=der_to_func,fprime2=double_der_to_func, disp=False)
        return E

def kepler_series_solution(M, e, terms=30):

    """Series solution of the kepler equation to reduce time
    """
    E = M
    for n in range(1, terms + 1):
        E += (2 / n) * jv(n, n * e) * np.sin(n * M)
    return E

def create_time_burst(M_phase,a_out,w_out,time_in,e,i_in,theta_s,phi_s,cycle):    

     t_outer = 2*np.pi*np.sqrt((np.power(a_out*au,3)/(G*4e6*solar_mass)))
     n = cycle * np.round(t_outer/time_in) + 1
     a= a_out*au
     n=np.round(n)
     n=int(n)
     T_burst=np.zeros(n)

     for i in range(n):

              t0=i*(time_in)
              M = t0 * np.sqrt((G * 4e6 * solar_mass / (np.power(a,3))))  + M_phase
              E = solve_kepler(M,e)  
              beta = e / (1 + np.sqrt(1 - e ** 2))
              F=E + 2 * np.arctan(beta * np.sin(E) / (1 - beta * np.cos(E)))
              R=(a * (1 - e** 2) / (1 + e * np.cos(F))) / C_CGS
              Doppler = 2 * np.pi * R * np.sin(F + w_out) * np.sin(i_in)
              t_calculated = Doppler / (2 * np.pi)
              t_calculated = (time_in*i) - t_calculated
          
              RD_LISA = 499.00478 * np.sin(theta_s) * np.cos((2 * np.pi * (i*time_in/T_lisa))-phi_s)
              T_burst[i] = t_calculated - RD_LISA
     T_burst-=T_burst[0]
     return T_burst,n
cycle=1.5

T_burst,n=create_time_burst(0,125,0.0,0.325e6,0.4,np.pi/4,np.pi/5,0,cycle)

print("It is for cycle\n",cycle)

print(n)
print(T_burst)

def to_minimize(constants):  
    """The cost function which should be minimized for
       the correct inference of the orbital parameters
    """

    M_phase=constants[0]  
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
    a*=1000
    time_in*=1e6
    M_out*=1e7

    for i in range(n):
# Roemer Delay due to revolution of Binary around the SBH

        t0=i*(time_in)
        M = t0 * np.sqrt((G * M_out * solar_mass / (np.power(a,3))))  + M_phase
        E = solve_kepler(M,e)  
        beta = e / (1 + np.sqrt(1 - e ** 2))
        F=E + 2 * np.arctan(beta * np.sin(E) / (1 - beta * np.cos(E)))
        R=(a * (1 - e** 2) / (1 + e * np.cos(F))) / C_CGS
        Doppler = 2 * np.pi * R * np.sin(F + w_out) * np.sin(i_in)
        t_calculated = Doppler / (2 * np.pi)
        t_calculated = (time_in*i) - t_calculated
  
# Roemer Delay due to the LISA's orbit

        RD_LISA = 499.00478 * np.sin(theta_s) * np.cos((2 * np.pi * (i*time_in/T_lisa))-phi_s)
        T_cal[i] = t_calculated - RD_LISA

    sigma=80
    T_cal = T_cal - T_cal[0]
    cost = np.sum((T_cal - T_burst) ** 2)/(2*(sigma**2))
    return cost     


bounds = [(-np.pi,np.pi),  
    (0.05, 0.4),
    (-np.pi,np.pi),     
    (0.0,0.85),    
    (0.0, 0.6),
    (-np.pi/2,np.pi/2),
    (-np.pi/2,np.pi/2),
    (-np.pi,np.pi)]

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
nwalkers = 384
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
nsteps = 40000

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

labels = ["M_{initial}","a_{out}","w_{out}","e_{out}","time_{in}","i","theta_source","phi_source"]

samples_array = np.array(samples)
file_path="Results_for_time_req"
if not os.path.exists(file_path):
    os.makedirs(file_path)

# Save the NumPy array to a text file in the specified directory
rname="1_5__cycle_mod_eccen"
file_path = os.path.join(file_path, rname+".txt")
np.savetxt(file_path, samples_array)

print(f'Samples saved to {file_path}')

#Saving the corner plot for this run
corner.corner(samples, labels=labels)
plt.savefig("Results_for_time_req/"+rname+".png")
