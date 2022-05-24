# Geometry paper
## General information
The repository contains data and codes complimentary to the paper on geometry estimation for fan-beam X-ray sawmill imaging. The folder containes synthetic data for two callibration phantoms: L-shaped calibration phantom (```./L_disk/```) and calibration phantom with a hole (```./hole_disk/```).

## Installation
The codes use the Operator Discretization Library (ODL), please see the installation guide: https://odlgroup.github.io/odl/getting_started/installing_source.html

## Geometry parameter estimation script
To estimate geometry parameters, one should run 

```python3 geom_param_search.py -d "./<calibration_phantom_name>/"```

where after option ```-d``` one should specify the folder with calibration phantom data. For example, in the current repository, two options are available:

```python3 geom_param_search.py -d "./L_disk/"```

to perform parameter search based on L-shaped calibration disk, or alternatively 

```python3 geom_param_search.py -d "./hole_disk/"```

to use calibratiom phantom with a hole. 

The result parameter vector will be saved as a numpy file to directory ```./<calibration_phantom_name>/params/```
