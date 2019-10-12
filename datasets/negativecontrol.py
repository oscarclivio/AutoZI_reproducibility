from scvi.dataset import GeneExpressionDataset
import os
import numpy as np

import anndata


class NegativeControlDataset(GeneExpressionDataset):

    def __init__(self, data, n_rna=100, threshold=None):

        super().__init__()
        self.n_rna = n_rna

        if isinstance(data, str):
            anndataset = anndata.read(data)
        else:
            anndataset = data

        expression_mat = np.array(anndataset.X.todense())

        # Eliminate zero cells and zero genes
        select_cells = expression_mat.sum(axis=1) > 0
        expression_mat = expression_mat[select_cells, :]
        select_genes = expression_mat.sum(axis=0) > 0
        expression_mat = expression_mat[:, select_genes]

        # Select ERCC genes
        gene_names = np.array([gene_name for idx, gene_name in enumerate(list(anndataset.var.index))
                               if select_genes[idx]])

        is_ercc_gene = np.array(['ercc' in gene_name.lower() for gene_name in gene_names])


        # Find top n_rna RNA expressed genes, mix ERCC and RNA genes
        is_rna_gene = ~is_ercc_gene
        rna_genes_indices = is_rna_gene.nonzero()[0]
        expression_mat_rna = expression_mat[:, is_rna_gene]
        top_genes = np.argsort(-expression_mat_rna.mean(axis=0))

        selected_rna_genes_indices = rna_genes_indices[top_genes[:n_rna]]
        ercc_genes_indices = is_ercc_gene.nonzero()[0]
        selected_genes = np.concatenate([selected_rna_genes_indices,ercc_genes_indices]).reshape(-1)
        np.random.shuffle(selected_genes)

        expression_mat = expression_mat[:, selected_genes]
        gene_names = gene_names[selected_genes]

        # Remove induced zero cells and keep only genes present in a given threshold of cells, if given
        select_cells = expression_mat.sum(axis=1) > 0
        expression_mat = expression_mat[select_cells, :]

        if threshold is not None:
            select_genes = (expression_mat > 0).mean(axis=0) >= threshold
        else:
            select_genes = (expression_mat > 0).mean(axis=0) > 0
        gene_names = gene_names[select_genes]
        expression_mat = expression_mat[:, select_genes]

        self.is_ercc = np.array(['ercc' in gene_name.lower() for gene_name in gene_names])

        self.populate_from_data(
            X=expression_mat,
            gene_names=gene_names,
        )


class Svensson1NegativeControlDataset(NegativeControlDataset):

    def __init__(self, n_rna=100, threshold=0.01):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(os.path.join(current_dir, 'svensson_chromium_control.h5ad'))

        sven1 = svens[svens.obs.query('sample == "20311"').index]
        super(Svensson1NegativeControlDataset, self).__init__(sven1,
                                                              n_rna=n_rna,
                                                              threshold=threshold)

class Svensson2NegativeControlDataset(NegativeControlDataset):

    def __init__(self, n_rna=100, threshold=0.01):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(os.path.join(current_dir, 'svensson_chromium_control.h5ad'))

        sven2 = svens[svens.obs.query('sample == "20312"').index]
        super(Svensson2NegativeControlDataset, self).__init__(sven2,
                                                              n_rna=n_rna,
                                                              threshold=threshold)

class KleinNegativeControlDataset(NegativeControlDataset):

    def __init__(self, n_rna=100, threshold=0.01):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        klein = anndata.read(os.path.join(current_dir, 'klein_indrops_control_GSM1599501.h5ad'))

        super(KleinNegativeControlDataset, self).__init__(klein,
                                                          n_rna=n_rna,
                                                          threshold=threshold)

class ZhengNegativeControlDataset(NegativeControlDataset):

    def __init__(self, n_rna=100, threshold=0.01):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        zheng = anndata.read(os.path.join(current_dir, 'zheng_gemcode_control.h5ad'))

        super(ZhengNegativeControlDataset, self).__init__(zheng,
                                                          n_rna=n_rna,
                                                          threshold=threshold)
