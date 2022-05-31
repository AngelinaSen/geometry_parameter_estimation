# Geometry paper
## General information
The repository contains data and codes complimentary to the paper on geometry estimation for fan-beam X-ray sawmill imaging. The folder containes synthetic data for two callibration phantoms: L-shaped calibration phantom (```./L_disk/```) and calibration phantom with a hole (```./hole_disk/```).

## Installation
The codes use the Operator Discretization Library (ODL), please see the installation guide: https://odlgroup.github.io/odl/getting_started/installing_source.html

## Geometry parameter estimation with calibration phantoms 

### Run one parameter search program 

To estimate geometry parameters, one should run 

```python3 geom_param_search.py -d "./<calibration_phantom_name>/"```

where after option ```-d``` one should specify the folder with calibration phantom data. For example, in the current repository, two options are available:

```python3 geom_param_search.py -d "./L_disk/"```

to perform parameter search based on L-shaped calibration disk, or alternatively 

```python3 geom_param_search.py -d "./hole_disk/"```

to use calibratiom phantom with a hole. 

The result parameter vector will be saved as a numpy file to directory ```./<calibration_phantom_name>/params/```

### Run several parameter search programs simultaniously 

There is a simple Bash-script in the repository that one can use to run N programs ```geom_param_search.py``` simultaniously.
For example, to execute 6 programs on geometry parameter search using the calibration phantom <calibration_phantom_name>, one should run

```bash param_search_runner.sh 6 "./<calibration_phantom_name>/"```

The output parameters will be saved to directory: ```./<calibration_phantom_name>/params/```

### Plot reconstructions with different parametrisation 

Aftre running at least 6 programs and getting 6 different parametrizations, one can plot them to compare. 

One can get the plot of reconstructions with different parametrizations  by running

```python3 plot_recos_with_params.py -d "./<calibration_phantom_name>/"```

## Geometry parameter estimation based on intersection points 
