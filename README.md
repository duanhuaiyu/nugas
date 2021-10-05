# NuGas: Computing Flavor Oscillations in Dense Neutrino Gases

`NuGas` is a Python package for computing flavor oscillations in dense neutrino gases. Please see the [Jupyter Notebooks](https://jupyter.org) under the directory `docs` for the examples that demonstrate its usage.

`NuGas` requires [Python3](https://www.python.org) and recent [`SciPy`](https://www.scipy.org/scipylib) and [`NumPy`](https://numpy.org) libraries. Some subpackages require [`pybind11`](https://pybind11.readthedocs.io/en/stable/) and [`netCDF4`](https://unidata.github.io/netcdf4-python/) packages and a modern C++ compiler that supports [OpenMP](https://www.openmp.org).

There are more than one way to install `NuGas`:
- If you are a casual user, try typing the following command in a terminal to install `NuGas`:  
`python3 -m pip install --upgrade git+https://github.com/NuCO-UNM/nugas.git`  
- If you are an advanced user and want to adapt the `NuGas` to your own needs, you can use `git` to clone the `NuGas` repository, `cd` into the `nugas` directory, and type the following command:   
`python3 -m pip install .`

`NuGas` is free to use under the MIT license. (See `LICENSE.txt` for details.)

This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Nuclear Physics under Award Number DE-SC-0017803.