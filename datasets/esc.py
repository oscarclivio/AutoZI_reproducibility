import pickle
import pandas as pd
import numpy as np
import anndata
from scvi.dataset.dataset import GeneExpressionDataset
import os
import tarfile



class KolodziejczykESCDataset(GeneExpressionDataset):
    def __init__(self, fraction_genes=0.1, fraction_uz_genes_left=0.1, kinetics_file_name='kinetics_c57_larsson_et_al_2019.csv', \
                 notations_dict_file_name="ensembl_to_symbol_dict.p", seed=0):
        super().__init__()

        np.random.seed(seed)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dict_notations = pickle.load(open(os.path.join(current_dir, notations_dict_file_name), "rb"))

        tar_path = os.path.join(current_dir, 'kolodziejczyk_et_al_2015_mesc.tar.gz')
        tar = tarfile.open(tar_path, "r:gz")
        tar.extractall(path=current_dir)

        anndata_kolod = anndata.read_h5ad(os.path.join(current_dir, 'kolodziejczyk_et_al_2015_mesc.h5ad'))
        genes_kolod = [dict_notations[ensembl_id] for ensembl_id in list(anndata_kolod.var.index) if
                       'ERCC' not in ensembl_id and len(dict_notations[ensembl_id]) > 0]
        valid_ensemb_ids = np.array([idx for (idx, ensembl_id) in enumerate(anndata_kolod.var.index) if
                                     'ERCC' not in ensembl_id and len(dict_notations[ensembl_id]) > 0])

        kinetics_df = pd.read_csv(os.path.join(current_dir, kinetics_file_name), index_col='Gene')
        kinetics_genes = list(kinetics_df.index)

        common_genes = np.array(sorted(list(set(genes_kolod).intersection(set(kinetics_genes)))))

        expr_mat_df = pd.DataFrame(data=anndata_kolod.X.toarray()[:, valid_ensemb_ids], columns=genes_kolod,
                                   index=anndata_kolod.obs.index)

        n_to_select = int(fraction_genes * len(common_genes))
        permutation = np.random.permutation(len(common_genes))

        expr_mat_df = expr_mat_df.loc[:, common_genes[permutation[:n_to_select]]]

        self.gene_labels = kinetics_df.loc[common_genes[permutation[:n_to_select]]]

        uz_genes = (self.gene_labels.regime_c57 == 'uz')
        n_to_select = int((fraction_uz_genes_left) * uz_genes.sum())
        uz_genes_to_keep = np.random.choice(uz_genes.index[uz_genes], n_to_select)
        uz_genes_to_remove = uz_genes
        uz_genes_to_remove[uz_genes_to_keep] = False

        self.gene_labels = self.gene_labels[~uz_genes_to_remove]
        expr_mat_df = expr_mat_df.loc[:, list(uz_genes.index[~uz_genes_to_remove])]


        self.populate_from_data(
            X=np.array(expr_mat_df),
            gene_names=np.array(expr_mat_df.columns),
        )

