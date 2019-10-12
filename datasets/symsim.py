from scvi.dataset import GeneExpressionDataset
import os
import numpy as np

import pandas as pd




class SymSimMixedDataset(GeneExpressionDataset):

    def __init__(self, ratios=(100,100,0), seed=0):

        if isinstance(ratios, list) or isinstance(ratios, tuple):
            ratios_dict = {}
            ratios_dict['b'] = ratios[0]
            ratios_dict['unz'] = ratios[1]
            ratios_dict['uz'] = ratios[2]
            ratios = ratios_dict

        np.random.seed(seed + ratios['b'] + ratios['uz'] + ratios['unz'])

        super().__init__()

        gene_expressions_true = {}
        gene_expressions_obs = {}
        meta = {}
        gene_labels = {}

        current_dir = os.path.dirname(os.path.realpath(__file__))

        for mode in ['b','unz','uz']:
            # Load
            gene_expressions_true[mode] = np.load(os.path.join(current_dir,\
                                                               'symsim_{mode}_singlepop_UMI_truecounts.npy')\
                                                  .format(mode=mode))
            gene_expressions_obs[mode] = np.load(os.path.join(current_dir, \
                                                               'symsim_{mode}_singlepop_UMI_obscounts.npy') \
                                                  .format(mode=mode))
            meta[mode] = pd.read_feather(os.path.join(current_dir, 'symsim_{mode}_singlepop_UMI_meta.feather')\
                                         .format(mode=mode))
            # Select desired proportion of genes
            n_to_select = int((ratios[mode] / 100.)*gene_expressions_true[mode].shape[1])
            permutation = np.random.permutation(gene_expressions_true[mode].shape[1])
            gene_expressions_true[mode] = gene_expressions_true[mode][:,permutation[:n_to_select]]
            gene_expressions_obs[mode] = gene_expressions_obs[mode][:,permutation[:n_to_select]]
            gene_labels[mode] = np.array([mode]*n_to_select)

        # Concatenation
        gene_expressions_final_true = np.concatenate((gene_expressions_true['b'],gene_expressions_true['unz'],\
                                                      gene_expressions_true['uz']),axis=1)
        gene_expressions_final_obs = np.concatenate((gene_expressions_obs['b'],gene_expressions_obs['unz'],\
                                                      gene_expressions_obs['uz']),axis=1)
        gene_labels_final = np.concatenate((gene_labels['b'], gene_labels['unz'], gene_labels['uz']))
        permutation = np.random.permutation(gene_expressions_final_true.shape[1])
        gene_expressions_final_true = gene_expressions_final_true[:,permutation]
        gene_expressions_final_obs = gene_expressions_final_obs[:,permutation]
        gene_labels_final = gene_labels_final[permutation]

        self.ratios = ratios
        self.meta = meta
        self.gene_expressions_true = gene_expressions_final_true
        self.gene_labels = gene_labels_final

        self.populate_from_data(
            gene_expressions_final_obs,
        )
