# AutoZI reproducibility repo

Reproduce the results from :
Oscar Clivio, Romain Lopez, Jeffrey Regier, Adam Gayoso, Michael I. Jordan, Nir Yosef . **Detecting Zero-Inflated Genes in Single-Cell Transcriptomics Data**. Submitted, 2019.
bioRxiv : https://www.biorxiv.org/content/10.1101/794875v2

## Installation

### scVI and AutoZI

- Clone the `autozi` branch of the scVI repo (https://github.com/YosefLab/scVI/tree/autozi) (to be merged with the `master` branch, WIP).
- Move to the root of the cloned repo and install scVI by running `python setup.py install`.

### AutoZI reproducibility code

In any folder of your choice.

## Reproducibility notebooks

### Background

In general, the following steps are performed in all notebooks:
1. Run AutoZI on datasets of interest;
2. Run built-in classification metrics on the outputs of AutoZI;
3. Apply 1. and 2. to the baseline procedures, when relevant;
4. Reproduce results for the experiment of interest.

All datasets used for analysis can be found in the `datasets` folder. The code for the baseline procedures can be found in `mle_nb.py`, `mle_zinb.py` and the notebooks for experiments where these baseline procedures were run : `poissonlognormal.ipynb` and `symsim.ipynb`. The code for classification metrics can be found in `classification_metrics.py`.

### Notebooks for experiments on synthetic datasets

1. Poisson log-normal datasets : `poissonlognormal.ipynb`
2. Beta-Poisson datasets : `symsim.ipynb`
3. Robustness of AutoZI's predictions for low/high average gene expression: `robustness.ipynb`

### Notebooks for experiments on real/biological datasets

1. Negative control datasets : `negativecontrol.ipynb`
2. 10X Genomics datasets: `10x.ipynb`
3. Mouse embyronic stem cell dataset : `esc.ipynb`