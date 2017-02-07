# Analyses for Waskom and Wagner (2017) **PNAS**

<img width=500px src=data/graphical_abstract.png>

This repository contains analysis code for the following paper:

Waskom M.L., Wagner A.D. (2017). Distributed representation of context by intrinsic subnetworks in prefrontal cortex. **Proceedings of the National Academy of the Sciences, USA**.

The paper can be accessed at the [PNAS website](http://www.pnas.org/cgi/doi/10.1073/pnas.1615269114).

A high-level overview of the commands that were executed to perform the analyses can be found in [`commands.md`](commands.md). Some of these commands execute [`lyman`](https://github.com/mwaskom/lyman) workflows using the parameters defined in the [`lyman/`](lyman/) subdirectory. Others execute experiment-specific code that are included in this repository. The experiment-specific code falls into a few different categories:

## Data extraction

- [`roi_cache.py`](roi_cache.py): Extracts timeseries data from the regions of interest that are analyzed in the paper.

## Analysis scripts

- [`decoding_analysis.py`](decoding_analysis.py): Performs the decoding analyses and estimation of context preferences.

- [`spatial_analysis.py`](spatial_analysis.py): Performs analyses of spatial distribution of context preferences.

- [`correlation_analysis.py`](correlation_analysis.py): Performs analyses relating to spontaneous correlations.

## Data compilation

- [`compile_data.py`](compile_data.py): Reads the individual outputs of the analysis scripts and compiles some summary data into tidy `.csv` files (stored in the [`data/`](data/) subdirectory).

## Statistical analyses

- [`paper_statistics.ipynb`](paper_statistics.ipynb): The statistical results reported in the paper were produced by analyses performed in this notebook.

## Figure scripts

These scripts generate each of the figures in the paper. They are contained in `.py` files with names corresponding to the relevant figure.

## Support libraries

- [`surfutils.py`](surfutils.py): Functions for going between volumetric and surface representations of data.

- [`plotutils.py`](plotutils.py): Functions that are useful for generating consistent figures.

## Version information

- [`lyman/environment.yml`](lyman/environment.yml): A `conda` environment file that should be able to reproduce all relevant software versions used for the analyses in the paper.
