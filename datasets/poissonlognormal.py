import logging
import os
import torch
from torch import distributions

import numpy as np

from scvi.dataset.dataset import (
    GeneExpressionDataset,
    DownloadableDataset,
    CellMeasurement,
)

logger = logging.getLogger(__name__)



class PoissonLogNormalDataset(GeneExpressionDataset):
    def __init__(self, pi=[0.7], loc_reduce=0, n_cells=100, mu0_path='mu_0.npy', mu1_path='mu_2.npy',
                 separate_reduce=False, sig0_path='sigma_0.npy', sig1_path='sigma_2.npy'):
        super().__init__()

        current_dir = os.path.dirname(os.path.realpath(__file__))
        mu_0 = self.load_array(os.path.join(current_dir, mu0_path))
        mu_1 = self.load_array(os.path.join(current_dir, mu1_path))
        sigma_0 = self.load_array(os.path.join(current_dir, sig0_path))
        sigma_1 = self.load_array(os.path.join(current_dir, sig1_path))

        np.random.seed(0)
        torch.manual_seed(0)

        n_genes = len(mu_0)

        if not(separate_reduce):
            self.dist0 = distributions.MultivariateNormal(loc=mu_0-loc_reduce, covariance_matrix=sigma_0)
            self.dist1 = distributions.MultivariateNormal(loc=mu_1-loc_reduce, covariance_matrix=sigma_1)
        else:

            n_genes *= 2

            mu_0_new = torch.zeros((2*mu_0.shape[0],))
            mu_0_new[:mu_0.shape[0]] = mu_0
            mu_0_new[-mu_0.shape[0]:] = mu_0 - loc_reduce
            mu_0_new = mu_0_new.double()

            mu_1_new = torch.zeros((2*mu_1.shape[0],))
            mu_1_new[:mu_1.shape[0]] = mu_1
            mu_1_new[-mu_1.shape[0]:] = mu_1 - loc_reduce
            mu_1_new = mu_1_new.double()

            sigma_0 = sigma_0.cpu().numpy()
            sigma_1 = sigma_1.cpu().numpy()
            fac = 4
            sigma_0 = sigma_0 / fac
            sigma_1 = sigma_1 / fac
            np.fill_diagonal(sigma_0, fac*np.diag(sigma_0))
            np.fill_diagonal(sigma_1, fac* np.diag(sigma_1))
            sigma_0 = torch.tensor(sigma_0).double()
            sigma_1 = torch.tensor(sigma_1).double()


            sigma_0_new = torch.zeros((2*sigma_0.shape[0],2*sigma_0.shape[1]))
            sigma_0_new[:sigma_0.shape[0], :sigma_0.shape[1]] = sigma_0
            sigma_0_new[sigma_0.shape[0]:, sigma_0.shape[1]:] = sigma_0
            sigma_0_new = sigma_0_new.double()

            sigma_1_new = torch.zeros((2 * sigma_1.shape[0], 2 * sigma_1.shape[1]))
            sigma_1_new[:sigma_1.shape[0], :sigma_1.shape[1]] = sigma_1
            sigma_1_new[sigma_1.shape[0]:, sigma_1.shape[1]:] = sigma_1
            sigma_1_new = sigma_1_new.double()

            self.dist0 = distributions.MultivariateNormal(loc=mu_0_new, covariance_matrix=sigma_0_new)
            self.dist1 = distributions.MultivariateNormal(loc=mu_1_new, covariance_matrix=sigma_1_new)


        cell_type = distributions.Bernoulli(probs=torch.tensor(pi)).sample((n_cells,))
        zero_mask = (cell_type == 0).squeeze()
        one_mask = (cell_type == 1).squeeze()

        z = torch.zeros((n_cells, n_genes)).double()

        z[zero_mask, :] = self.dist0.sample((zero_mask.sum(),))
        z[one_mask, :] = self.dist1.sample((one_mask.sum(),))

        gene_expressions = distributions.Poisson(rate=z.exp()).sample().cpu().numpy()
        labels = cell_type.cpu().numpy()

        self.mask_zero_biological = (gene_expressions == 0)


        gene_expressions, batches = self.mask(gene_expressions, labels)

        gene_names = np.arange(n_genes).astype(str)

        keep_cells = (gene_expressions.sum(axis=1) > 0)
        gene_expressions = gene_expressions[keep_cells,:]
        if labels is not None:
            labels = labels[keep_cells]
        if batches is not None:
            batches = batches[keep_cells]

        self.populate_from_data(
            gene_expressions,
            labels =labels,
            gene_names=gene_names,
            batch_indices=batches,
        )

    @staticmethod
    def load_array(path):
        return torch.tensor(np.load(path))

    def mask(self, data, labels):
        return data, None

class ZIFAPoissonLogNormalDataset(PoissonLogNormalDataset):
    def __init__(self, dropout_coef=0.08, dropout_lambda=1e-3, **kwargs):
        self.dropout_coef = dropout_coef
        self.dropout_lambda=dropout_lambda
        super(ZIFAPoissonLogNormalDataset, self).__init__(**kwargs)

    def mask(self, data, labels):
        self.dropout = self.dropout_coef * np.exp(-self.dropout_lambda * data**2)
        mask = np.random.binomial(1, 1 - self.dropout).astype(bool)
        self.mask_zero_zi = ~mask

        return data * mask, None

class ZIFAPoissonLogNormalDatasetMixed(PoissonLogNormalDataset):
    def __init__(self, dropout_coef=0.08, dropout_lambda=1e-3, zero_inflation_share=0.5, n_batches_tot=None,**kwargs):
        self.dropout_coef = dropout_coef
        self.dropout_lambda=dropout_lambda
        self.zero_inflation_share = zero_inflation_share
        self.n_batches_tot = n_batches_tot
        super(ZIFAPoissonLogNormalDatasetMixed, self).__init__(**kwargs)

    def mask(self, data, labels):

        self.dropout = self.dropout_coef * np.exp(-self.dropout_lambda * data**2)
        mask = np.random.binomial(1, 1 - self.dropout).astype(bool)

        no_zi_genes = np.ones((mask.shape[1],)).astype(bool)
        no_zi_genes[:int(self.zero_inflation_share*no_zi_genes.size)] = False
        np.random.shuffle(no_zi_genes)
        mask[:,no_zi_genes] = True

        self.no_zi_genes = no_zi_genes
        self.mask_zero_zi = ~mask

        if self.n_batches_tot is not None:
            batch_size = int(mask.shape[0] / self.n_batches_tot)
            batches = []
            self.no_zi_genes_batches = np.zeros((mask.shape[1], self.n_batches_tot)).astype(bool)
            for batch_index in range(self.n_batches_tot):
                ind_min = batch_index * batch_size
                ind_max = min((batch_index + 1) * batch_size, mask.shape[0])
                batches += [batch_index] * (ind_max - ind_min)
                self.no_zi_genes_batches[:, batch_index] = self.no_zi_genes
            batches = np.array(batches)
        else:
            batches = None
            self.no_zi_genes_batches = None

        return data * mask, batches
