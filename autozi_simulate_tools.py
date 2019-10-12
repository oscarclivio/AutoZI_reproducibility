import torch
import pickle
import argparse
import re
import numpy as np
import time



def retrieve_rates_dropouts(vae, posterior, n_samples=100):
    px_rates = []
    px_dropout_probs = []
    for tensors in posterior:
        sample_batch, _, _, batch_index, labels = tensors
        outputs = vae.inference(sample_batch,batch_index=batch_index,y=labels,n_samples=n_samples)
        px_rate = outputs["px_rate"]
        px_dropout = outputs["px_dropout"]
        if n_samples > 1:
            px_rate = torch.mean(px_rate,dim=0)
            px_dropout = torch.mean(px_dropout,dim=0)
        px_dropout_probs_batch = 1. / (1. + torch.exp(-px_dropout))
        px_dropout_probs_batch = px_dropout_probs_batch.detach().cpu().numpy()
        px_dropout_probs.append(px_dropout_probs_batch)
        px_rates.append(px_rate.detach().cpu().numpy())
    px_rates = np.concatenate(px_rates)
    px_rates_all = np.array(px_rates)
    px_rates = px_rates.mean(axis=0)

    px_dropout_probs = np.concatenate(px_dropout_probs)
    px_dropout_probs = px_dropout_probs.mean(axis=0)
    return px_dropout_probs, px_rates, px_rates_all




def retrieve_rates_dropouts_per_cell_type(vae, posterior, n_samples=100):
    px_dropout_probs = []
    px_rates = []
    labels_all = []
    for tensors in posterior:
        sample_batch, _, _, batch_index, labels = tensors
        outputs = vae.inference(sample_batch, batch_index=batch_index, y=labels, n_samples=n_samples)
        px_rate = outputs["px_rate"]
        px_dropout = outputs["px_dropout"]
        if n_samples > 1:
            px_rate = torch.mean(px_rate, dim=0)
            px_dropout = torch.mean(px_dropout,dim=0)
        labels_all.append(labels.cpu().numpy().reshape(-1))
        px_dropout_probs_batch = 1. / (1. + torch.exp(-px_dropout))
        px_dropout_probs_batch = px_dropout_probs_batch.detach().cpu().numpy()
        px_dropout_probs.append(px_dropout_probs_batch)
        px_rates.append(px_rate.detach().cpu().numpy())
    labels_all = np.concatenate(labels_all)
    px_dropout_probs = np.concatenate(px_dropout_probs)
    px_rates = np.concatenate(px_rates)
    px_rates_all = np.array(px_rates)
    px_dropout_probs_dict = {}
    px_rates_dict = {}
    for label in np.unique(labels_all):
        px_rates_dict[label] = px_rates[labels_all == label,:].mean(axis=0)
        px_dropout_probs_dict[label] = px_dropout_probs[labels_all == label,:].mean(axis=0)
    return px_dropout_probs_dict, px_rates_dict, px_rates_all



def restrict_to_common_nonzero_genes_cell_types(tenxv1, tenxv2, tenxv3, threshold=0.):
    common_genes = set(tenxv1.gene_names).intersection(set(tenxv2.gene_names).intersection(set(tenxv3.gene_names)))


    genes_to_study = {}

    for label in np.unique(tenxv1.cell_types):
        nonzero_genes_label = common_genes
        enoughexp_genes_label = common_genes
        genes_to_study[label] = []

        for tenx in [tenxv1, tenxv2, tenxv3]:
            label_ind_dataset = tenx.cell_types_to_labels([label])[0]
            zero_genes_idx_label_dataset = (tenx.X.toarray()[(tenx.labels == label_ind_dataset).flatten(), :]\
                                            .mean(axis=0) <= threshold).flatten()
            nonzero_genes_label = nonzero_genes_label.intersection(tenx.gene_names[~zero_genes_idx_label_dataset])

            enoughexp_genes_idx_label_dataset = (tenx.X.toarray()[(tenx.labels == label_ind_dataset).flatten(), :] \
                                            .mean(axis=0) > 1.).flatten()
            enoughexp_genes_label = enoughexp_genes_label.intersection(\
                tenx.gene_names[enoughexp_genes_idx_label_dataset])

        for tenx in [tenxv1, tenxv2, tenxv3]:
            genes_to_study[label].append(enoughexp_genes_label)

        common_genes = common_genes.intersection(nonzero_genes_label)


    common_genes = list(common_genes)
    for tenx in [tenxv1, tenxv2, tenxv3]:
        tenx.filter_genes_by_attribute(common_genes)
        tenx.filter_cells_by_count()

    genes_to_study_idx = {}
    for label in genes_to_study:
        genes_to_study_idx[label] = [tenx.genes_to_index(list(genes.intersection(set(common_genes))))\
                                                         for genes in genes_to_study[label]]

    return genes_to_study_idx
