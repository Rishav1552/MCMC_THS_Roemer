
## MCMC Sampler on time series data of GW burst from Triple Hierarchical System

The code allows the estimation of eigth orbital parameters given the GW burst data from the system. It uses roemer delay in the time data of these bursts to infer these parameters:

i. Argument of Pericenter of outer orbit
ii. Inclination of outer orbit
iii. Semi-major axis of outer orbit
iv. Eccentricity of outer orbit
v. Time period of inner orbit 
vi. Initial mean anomaly of outer orbit
vii. Polar angle of source with respect to LISA
viii. Azimuthal angle of source with respect to LISA

## Installation

Create a separate python environment with python 3.10. If you are using conda you can create a env named mcmc like this:
```bash
  conda create --name mcmc python=3.10
```

Clone the repository at the desired location:

```bash
  git clone https://github.com/Rishav1552/MCMC_THS_Roemer.git
```
Install the necessary libraries and packages
```bash
  pip install -r requirements.txt
```
Run the code

i. For MCMC sampler with your own time series and index data
```bash
  python MCMC.py path_to_time_series_data path_index_data output_filename
```
i. Data from the same model as used in this code and with desired cycle od doppler modulation
```bash
  python MCMC_check.py cycle
```


## Documentation

### Directories:

i. time_data: Includes the time series of GW burst data

ii. indices: Includes the indices of the time series of GW burst data

iii. Results with source localization: The output files (.txt file of MCMC sampler chain and corner plot) are stored here. It stores the output from 

iv. Results_for_time_req: To store the output from MCMC_check.py

### Files:

i. MCMC.py: runs the MCMC sampler given the time series and index data.

ii. MCMC_check.py: This code checks the ability of MCMC sampler to get the correct value given a desired cycle of doppler modulation.



